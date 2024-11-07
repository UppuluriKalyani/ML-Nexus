import torch
import torch.nn as nn
from fairseq_user.pretrained_enc.dino.vits import vit_large, vit_base

class Encoder(nn.Module):
    def __init__(self, model_path, layer) -> None:
        super().__init__()        
        self.layer = layer
        
        self.vit = vit_base(14, 4, block_chunks=0, init_values=1)
        state_dict = torch.load(model_path, map_location="cpu")
        self.vit.load_state_dict(state_dict, strict=True)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
            
    @torch.inference_mode()
    def forward(self, img):
        
        h, w = img.shape[-2:]
        
        rep = self.vit.get_intermediate_layers(img, self.layer, return_all=False)[0]
        
        return rep
    
