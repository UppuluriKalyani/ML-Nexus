# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from torch import Tensor
from typing import Any, Dict, List, Optional
import copy
import torch
import torch.nn as nn
import math

from omegaconf import II
from dataclasses import dataclass, field

from fairseq.models import (
    FairseqLanguageModel,
    register_model,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
from fairseq.modules import PositionalEmbedding
import numpy as np

from fairseq_user.pretrained_enc import models_pretrained_enc as models_pretrained_enc
from .taming.models.vqgan import VQModel

logger = logging.getLogger(__name__)

@dataclass
class PretrainedEncConfig:
    pretrained_enc_arch: Optional[str] = False
    pretrained_enc_path: Optional[str] = False
    pretrained_enc_proj_dim: Optional[int] = None
    pretrained_enc_withproj: bool = True
    layer: int = 3
    
@dataclass
class ST2VTLMConfig(TransformerLanguageModelConfig):
    decoder_normalize_before: bool = True
    vqgan_ckpt_path: Optional[str] = None
    vqgan_config_path: Optional[str] = None
    pretrained_enc_config: Optional[PretrainedEncConfig] = None
    kmeans_path: Optional[str] = None
    base_shuffle: bool = False
    
class TransformerDecoder2pos(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.embed_positions_1 = (
            PositionalEmbedding(
                self.max_target_positions,
                self.embed_dim,
                self.padding_idx,
                learned=self.cfg.decoder.learned_pos,
            )
            if not self.cfg.no_token_positional_embeddings
            else None
        )
        
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        id_eos = torch.nonzero(prev_output_tokens[0].eq(self.dictionary.eos()))
        if self.embed_positions is not None:
            if incremental_state is None:
                positions_1 = self.embed_positions_1(
                    prev_output_tokens[:,:id_eos], incremental_state=incremental_state
                )
                positions_2 = self.embed_positions(
                    prev_output_tokens[:,id_eos:], incremental_state=incremental_state
                )
                positions = torch.cat([positions_1, positions_2], dim = 1)
            else:
                # the prfix won't be here
                positions = self.embed_positions(
                    prev_output_tokens, incremental_state=incremental_state
                )
        
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model("st2vtlm", dataclass=ST2VTLMConfig)
class ST2VTTransformerLanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder, vqgan, pretrained_encoder, centroids):
        super().__init__(decoder)
        self.args = args
        self.vqgan = vqgan
        self.pretrained_encoder = pretrained_encoder
        self.centroids = nn.Parameter(centroids, requires_grad=False)
        self.centroid_norm = nn.Parameter((centroids ** 2).sum(0, keepdims=True), requires_grad=False)
  
    def build_emb(cls, args, task):
        embed_tokens = Embedding(
            len(task.target_dictionary) + len(task.source_dictionary), 
            args.decoder_embed_dim, 
            task.target_dictionary.pad()
        )
        return embed_tokens
    
    @classmethod
    def build_model(cls, args, task):    
                
        embed_tokens = cls.build_emb(cls, args, task)
        vqgan = cls.build_vqgan(cls, args)
        pretrained_encoder = cls.build_pre_trained_encoder(cls, args)
        centroids = cls.build_centroids(cls, args)
        
        decoder = TransformerDecoder2pos(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
            
        return cls(args, decoder, vqgan, pretrained_encoder, centroids)
    
    def build_pre_trained_encoder(cls, args):
        pretrained_enc_config = args.pretrained_enc_config
        assert pretrained_enc_config.pretrained_enc_path is not None
        assert pretrained_enc_config.pretrained_enc_arch is not None
        
        pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_config.pretrained_enc_arch](proj_dim=pretrained_enc_config.pretrained_enc_proj_dim)
        # load pre-trained encoder parameters
        if 'moco' in pretrained_enc_config.pretrained_enc_arch:
            pretrained_encoder = models_pretrained_enc.load_pretrained_moco(pretrained_encoder,
                                                                                 pretrained_enc_config.pretrained_enc_path)
        elif 'dino' in pretrained_enc_config.pretrained_enc_arch:
            pretrained_encoder = models_pretrained_enc.load_pretrained_dino(pretrained_encoder,
                                                                                 pretrained_enc_config.pretrained_enc_path)
        elif 'ibot' in pretrained_enc_config.pretrained_enc_arch:
            pretrained_encoder = models_pretrained_enc.load_pretrained_ibot(pretrained_encoder,
                                                                                 pretrained_enc_config.pretrained_enc_path)
        elif 'mae' in pretrained_enc_config.pretrained_enc_arch:
            pretrained_encoder = models_pretrained_enc.load_pretrained_mae(pretrained_encoder,
                                                                                 pretrained_enc_config.pretrained_enc_path)
        else:
            raise NotImplementedError

        for param in pretrained_encoder.parameters():
            param.requires_grad = False
            
        return pretrained_encoder
    
    def build_centroids(cls, args):
        centroids = np.load(args.kmeans_path)
        centroids = torch.from_numpy(centroids).transpose(0,1).contiguous()
        return centroids
    
    def build_vqgan(cls, args): 
        from omegaconf import OmegaConf
        vqgan_config = OmegaConf.load(args.vqgan_config_path).model
        vqgan = VQModel(ddconfig=vqgan_config.params.ddconfig,
                             n_embed=vqgan_config.params.n_embed,
                             embed_dim=vqgan_config.params.embed_dim,
                             ckpt_path=args.vqgan_ckpt_path)
        for param in vqgan.parameters():
            param.requires_grad = False
            
        return vqgan
    
    def add_eos_shift(self, x, sizes):
        x = x + self.decoder.dictionary.nspecial
        
        x_src = torch.cat([x.new(x.shape[0],1).fill_(self.decoder.dictionary.eos()), x], dim = 1)
        x_tgt = torch.cat([x, x.new(x.shape[0],1).fill_(self.decoder.dictionary.eos())], dim = 1)
    
        return x_src, x_tgt
    
    @torch.no_grad()
    def prepare_semantic(self, imgs, img_sizes):
        self.pretrained_encoder.eval()
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(imgs.dtype)
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(imgs.dtype)
        x_normalized = (imgs - mean) / std
        if x_normalized.shape[-1] == 256:
            x_normalized = torch.nn.functional.interpolate(x_normalized, scale_factor=224 / 256, mode='bicubic').to(imgs.dtype)
        else:
            assert x_normalized.shape[-1] == 224
        rep = self.pretrained_encoder.get_intermediate_layers(x_normalized, self.args.pretrained_enc_config.layer)[0]
        
        dist = (
            rep.flatten(0,1).pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(rep.flatten(0,1), self.centroids)
            + self.centroid_norm
        )
        sem = dist.argmin(dim=-1)
        sem = sem.view(rep.shape[:2])
        
        sem = sem + len(self.decoder.dictionary) + self.decoder.dictionary.nspecial
            
        return sem
    
    def prepare_incremental_state(
            self,
            net_input,
            incremental_state,
            beam_size=1,
        ): # compute semantic kv cache at once 

        imgs = net_input["imgs"] if net_input["imgs"] is not None else net_input["src_tokens"]
        img_sizes = net_input["img_sizes"]
        
        if imgs.dim() == 4:
            src_tokens = self.prepare_semantic(imgs.type_as(self.decoder.embed_tokens.weight), img_sizes)
        elif imgs.dim() == 2:
            src_tokens = imgs + len(self.decoder.dictionary)
        else:
            raise NotImplementedError
        
        if beam_size != 1:
            src_tokens = src_tokens.repeat_interleave(beam_size, dim=0)
        
        s_embeds = self.decoder.embed_scale * self.decoder.embed_tokens(src_tokens)
        positions = self.decoder.embed_positions_1(src_tokens)
        x = s_embeds + positions
        
        alignment_layer = self.decoder.num_layers - 1
        if self.decoder.layernorm_embedding is not None:
            x = self.decoder.layernorm_embedding(x)
        x = self.decoder.dropout_module(x)
        x = x.transpose(0, 1)
        self_attn_padding_mask = None
        if self.decoder.cross_self_attention or src_tokens.eq(self.decoder.padding_idx).any():
            self_attn_padding_mask = src_tokens.eq(self.decoder.padding_idx)
        for idx, layer in enumerate(self.decoder.layers):
            self_attn_mask = self.decoder.buffered_future_mask(x)
            x, layer_attn, _ = layer(
                x,
                None,
                None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
        
        return None

        
    def forward(
        self,
        sample,
        **kwargs
    ):
        imgs = sample["net_input"]["imgs"]
        img_sizes = sample["net_input"]["img_sizes"]
        
        sem = self.prepare_semantic(imgs, img_sizes)
            
        with torch.no_grad():
            #self.vqgan.eval()
            z_q, _, token_tuple = self.vqgan.encode(imgs)
            _, _, token_indices = token_tuple
            token_indices = token_indices.reshape(z_q.size(0), -1)
            src, tgt = self.add_eos_shift(token_indices, img_sizes // 16)
                 
        decoder_input = torch.cat([sem, src], dim = 1)
        logits, extra = self.decoder(
            decoder_input,
        )
        
        logits = logits[:, sem.shape[1]:]
        
        output = {
            "word_ins_visual": {
                "out": logits,
                "mask": src.ne(self.decoder.padding_idx), # token level
                "nll_loss": True,
                "factor": 1.0,
                "tgt": tgt
            },
        }
        
        return output, extra
    