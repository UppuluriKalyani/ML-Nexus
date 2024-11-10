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

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    FairseqIncrementalDecoder,
)
from fairseq.utils import safe_getattr, safe_hasattr, get_available_activation_fns
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)
from fairseq.models.transformer_lm import TransformerLanguageModelConfig
import numpy as np

from fairseq_user.pretrained_enc import models_pretrained_enc as models_pretrained_enc


logger = logging.getLogger(__name__)

@dataclass
class PretrainedEncConfig:
    pretrained_enc_arch: Optional[str] = False
    pretrained_enc_path: Optional[str] = False
    pretrained_enc_proj_dim: Optional[int] = None
    pretrained_enc_withproj: bool = True
    layer: int = 3
    
@dataclass
class STLMConfig(TransformerLanguageModelConfig):
    decoder_normalize_before: bool = False
    pretrained_enc_config: Optional[PretrainedEncConfig] = None
    kmeans_path: Optional[str] = None
    class_conditional: bool = False  
    pretrained_decoder_path:  Optional[str] = None    

@register_model("stlm_linearprobe", dataclass=STLMConfig)
class STTransformerLanguageModelLinearProbe(FairseqLanguageModel):
    def __init__(self, args, decoder, pretrained_encoder, centroids):
        super().__init__(decoder)
        self.args = args
        self.pretrained_encoder = pretrained_encoder
        self.centroids = nn.Parameter(centroids, requires_grad=False)
        self.centroid_norm = nn.Parameter((centroids ** 2).sum(0, keepdims=True), requires_grad=False)

        self.linear_classifiers = nn.ModuleDict()
        for n in range(args.decoder_layers):
            linear_classifier = nn.Linear(args.decoder_embed_dim, 1000)
            linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
            linear_classifier.bias.data.zero_()

            self.linear_classifiers.update({
                f"classifier_{n}_blocks".replace(".", "_"): linear_classifier
            })
        
    def build_emb(cls, args, task):
        embed_tokens = Embedding(
            len(task.target_dictionary) if not args.class_conditional else len(task.target_dictionary) + 1000, 
            args.decoder_embed_dim, 
            task.target_dictionary.pad()
        )
        return embed_tokens
    
    @classmethod
    def build_model(cls, args, task):    
                
        embed_tokens = cls.build_emb(cls, args, task)
        pretrained_encoder = cls.build_pre_trained_encoder(cls, args)
        centroids = cls.build_centroids(cls, args)
        
        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
            
        state_dict = torch.load(args.pretrained_decoder_path, map_location="cpu")

        new_state_dict = {}
        for k in list(state_dict['model'].keys()):
            if k.startswith('decoder'):
                new_k = k[len("decoder."):]
                new_state_dict[new_k] = state_dict['model'][k]
        del state_dict
        decoder.load_state_dict(new_state_dict, strict=True)

        for param in decoder.parameters():
            param.requires_grad = False
        
        return cls(args, decoder, pretrained_encoder, centroids)
    
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
    
    def add_eos_shift(self, x, sizes):
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
        
        sem = sem + self.decoder.dictionary.nspecial
        return sem
    
    def forward(
        self,
        sample,
        **kwargs
    ):
        imgs = sample["net_input"]["imgs"]
        img_sizes = sample["net_input"]["img_sizes"]

        sem = self.prepare_semantic(imgs, img_sizes)
        src, tgt = self.add_eos_shift(sem, (img_sizes * (224 / 256)).long() // 14)

        if self.args.class_conditional:
            src[:,0] = sample["net_input"]["cls_ids"] + len(self.decoder.dictionary)
                
        with torch.no_grad():
            self.decoder.eval()
            logits, extra = self.decoder(
                src,
                features_only=True,
            )

        output = {}
        for n in range(self.args.decoder_layers):
            k = f"classifier_{n}_blocks".replace(".", "_")
            output[k] = {
                "out": self.linear_classifiers[k](extra["inner_states"][n].mean(0).detach()),
                "mask": None,
                "nll_loss": True,
                "factor": 1.0,
                "tgt": sample["net_input"]["cls_ids"]
            }
                
       
        return output, extra
