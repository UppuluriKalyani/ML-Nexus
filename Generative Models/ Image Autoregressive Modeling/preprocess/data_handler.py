import logging
from torchvision import datasets
import os
from torchvision import transforms as pth_transforms
import torch

logger = logging.getLogger(__name__)


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train"):
        super().__init__()
                
        self.dataset = datasets.ImageFolder(
                os.path.join(root, split),
            )
        
        self.transform = pth_transforms.Compose([
            pth_transforms.Resize(224, interpolation=3),
            pth_transforms.RandomCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        logger.info(
            f"Init dataset with root in {root}, containing {len(self.dataset)} files"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):            
        img, _ = self.dataset[index]
        img = self.transform(img)
        
        return img


