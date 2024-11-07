import torch.optim

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from collections.abc import Collection
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf
from typing import Any, List


@dataclass
class FairseqSGDConfig(FairseqDataclass):
    momentum: float = field(
        default=0.0
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    lr: List[float] = II("optimization.lr")
   


@register_optimizer("sgd_hydra", dataclass=FairseqSGDConfig)
class SGD(FairseqOptimizer):
    def __init__(self, cfg, params):
        super().__init__(cfg)
        self._optimizer = torch.optim.SGD(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "momentum": self.cfg.momentum,
            "weight_decay": self.cfg.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True