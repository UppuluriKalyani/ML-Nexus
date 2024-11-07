import torch
import torch.nn as nn
from .moco_v3 import vits as moco_vits
from .dino import vits as dino_vits
from .ibot import vits as ibot_vits
from .mae import vits as mae_vits

def build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

def load_pretrained_moco(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    return model

def load_pretrained_mae(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=True)
    return model

def load_pretrained_dino(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    return model

def load_pretrained_ibot(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # rename ibot pre-trained keys
    state_dict = checkpoint['teacher']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('backbone'):
            # remove prefix
            state_dict[k[len("backbone."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    del state_dict['head.last_layer.weight_g']
    del state_dict['head.last_layer.weight_v']
    del state_dict['head.last_layer2.weight_g']
    del state_dict['head.last_layer2.weight_v']
    model.load_state_dict(state_dict, strict=True)
    return model


def dino_vit_base(proj_dim, **kwargs):
    model = dino_vits.vit_base(14, 4, block_chunks=0, init_values=1)
    
    return model

def dino_vit_large(proj_dim, **kwargs):
    model = dino_vits.vit_large(14, 4, block_chunks=0, init_values=1)
    
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


def ibot_vit_base(proj_dim, **kwargs):    
    model = ibot_vits.vit_base()
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    model.head = DINOHead(in_dim=hidden_dim, bottleneck_dim=proj_dim)
    return model

def mae_vit_base(proj_dim, **kwargs):
    model = mae_vits.mae_vit_base_patch16()
    del model.decoder_blocks, model.decoder_embed, model.decoder_pos_embed, model.decoder_norm, model.decoder_pred, model.mask_token

    return model

def mocov3_vit_base(proj_dim, **kwargs):
    model = moco_vits.vit_base(**kwargs)
    hidden_dim = model.head.weight.shape[1]
    del model.head  # remove original head layer

    # projectors
    model.head = build_mlp(3, hidden_dim, 4096, proj_dim)
    return model
