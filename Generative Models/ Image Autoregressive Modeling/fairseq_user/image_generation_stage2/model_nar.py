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
from functools import partial

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
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
from fairseq.modules import PositionalEmbedding
from fairseq.utils import new_arange
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


@register_model("st2vtlm_nar", dataclass=ST2VTLMConfig)
class ST2VTTransformerLanguageModelNAR(FairseqLanguageModel):
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
    
    def random_mask(self, x, sizes):
        target_tokens = x + self.decoder.dictionary.nspecial
        target_masks = target_tokens.ne(self.decoder.dictionary.eos())    # eos
        target_score = target_tokens.clone().float().uniform_() # cosine
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * torch.cos(target_length.clone().uniform_() * math.pi / 2)
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), self.decoder.dictionary.unk() # 3 is unk
        )
        
        x_src = torch.cat([prev_target_tokens.new(x.shape[0],1).fill_(self.decoder.dictionary.eos()), prev_target_tokens], dim = 1)
        x_tgt = torch.cat([target_tokens.new(x.shape[0],1).fill_(self.decoder.dictionary.eos()), target_tokens], dim = 1)
    
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

    def initialize_output_tokens(self, net_input):
        
        imgs = net_input["imgs"] if net_input["imgs"] is not None else net_input["src_tokens"]
        img_sizes = net_input["img_sizes"]
        
        if imgs.dim() == 4:
            src_tokens = self.prepare_semantic(imgs, img_sizes)
        elif imgs.dim() == 2:
            src_tokens = imgs + len(self.decoder.dictionary)
        else:
            raise NotImplementedError

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), 257
        ).fill_(self.decoder.dictionary.unk())
        initial_output_tokens[:, 0] = self.decoder.dictionary.eos()
        initial_output_tokens = torch.cat([src_tokens, initial_output_tokens], dim = 1)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(self.decoder.embed_tokens(initial_output_tokens))
    
        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def forward(
        self,
        sample,
        **kwargs
    ):
        imgs = sample["net_input"]["imgs"]
        img_sizes = sample["net_input"]["img_sizes"]
        
        sem = self.prepare_semantic(imgs, img_sizes)
            
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(imgs)
            _, _, token_indices = token_tuple
            token_indices = token_indices.reshape(z_q.size(0), -1)
            src, tgt = self.random_mask(token_indices, img_sizes // 16)
                 
        decoder_input = torch.cat([sem, src], dim = 1)
        logits, extra = self.decoder(
            decoder_input,
            full_context_alignment=True,
        )
        
        logits = logits[:, sem.shape[1]:]
        
        output = {
            "word_ins_visual": {
                "out": logits,
                "mask": src.eq(self.decoder.dictionary.unk()), # unk part
                "nll_loss": True,
                "factor": 1.0,
                "tgt": tgt
            },
        }
        
        return output, extra
    
    
    def forward_decoder(self, decoder_out, decoding_format=None, **kwargs):
        
        unknown_number_in_the_beginning = 256
        _CONFIDENCE_OF_KNOWN_TOKENS = math.inf
        choice_temperature = 4.5
        temp = 1.0

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.decoder.dictionary.unk())
        logits, _ = self.decoder(
            output_tokens,
            full_context_alignment=True,
        )

        logits[...,:self.decoder.dictionary.nspecial] = -_CONFIDENCE_OF_KNOWN_TOKENS

        if history is not None:
            history.append(output_tokens.clone())

        ratio = 1. * (step + 1) / max_step
        
        sample_dist = torch.distributions.categorical.Categorical(logits=logits / temp)
        sampled_ids = sample_dist.sample()
        output_tokens = torch.where(output_masks, sampled_ids, output_tokens)
        assert output_tokens.ne(self.decoder.dictionary.unk()).all()
        

        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(output_tokens.clip(max=len(self.decoder.dictionary) -1), -1)), -1)
        

        selected_scores = torch.squeeze(
            torch.gather(torch.log(probs), dim=-1, index=torch.unsqueeze(output_tokens.clip(max=len(self.decoder.dictionary) - 1), -1)), -1)
        output_scores = torch.where(output_masks, selected_scores, output_scores)
        

        if (step + 1) != max_step:
            mask_ratio = np.cos(math.pi / 2. * ratio)
            selected_probs = torch.where(output_masks, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS).type_as(logits)
     
            mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                    torch.minimum(torch.sum(output_masks, dim=-1, keepdims=True) - 1, mask_len))
           
            skeptical_mask = mask_by_random_topk(mask_len[0], selected_probs, choice_temperature * (1 - ratio))
            output_tokens.masked_fill_(skeptical_mask, self.decoder.dictionary.unk())
            output_scores.masked_fill_(skeptical_mask, 0.0)

        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + torch.Tensor(temperature * np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, axis=-1)
    # Obtains cut off threshold given the mask lengths.
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    # Masks tokens with lower confidence.
    masking = (confidence <= cut_off)
    return masking
