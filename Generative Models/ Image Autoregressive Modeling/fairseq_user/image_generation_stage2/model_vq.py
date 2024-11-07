# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from torch import Tensor
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

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
from .taming.models.vqgan import VQModel


logger = logging.getLogger(__name__)


@dataclass
class ST2VTLMConfig(TransformerLanguageModelConfig):
    vqgan_ckpt_path: Optional[str] = None
    vqgan_config_path: Optional[str] = None
    decoder_normalize_before: bool = True
    base_shuffle: bool = False
    
@register_model("st2vtlm_vq", dataclass=ST2VTLMConfig)
class ST2VTTransformerLanguageModelVQ(FairseqLanguageModel):
    def __init__(self, args, decoder, vqgan):
        super().__init__(decoder)
        self.args = args
        self.vqgan = vqgan
 
    def build_emb(cls, args, task):
        embed_tokens = Embedding(
            len(task.target_dictionary), 
            args.decoder_embed_dim, 
            task.target_dictionary.pad()
        )
        return embed_tokens
    
    @classmethod
    def build_model(cls, args, task):    
                
        embed_tokens = cls.build_emb(cls, args, task)
        vqgan = cls.build_vqgan(cls, args)
        
        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
            
        return cls(args, decoder, vqgan)

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
    
    def prepare_incremental_state(
            self,
            net_input,
            incremental_state,
            beam_size,
        ):

        return None

    
    def forward(
        self,
        sample,
        **kwargs
    ):
        imgs = sample["net_input"]["imgs"]
        img_sizes = sample["net_input"]["img_sizes"]
        
        with torch.no_grad():
            #self.vqgan.eval()
            z_q, _, token_tuple = self.vqgan.encode(imgs)
            _, _, token_indices = token_tuple
            token_indices = token_indices.reshape(z_q.size(0), -1)
            src, tgt = self.add_eos_shift(token_indices, img_sizes // 16)
                 
        logits, extra = self.decoder(
            src,
        )
                
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
    