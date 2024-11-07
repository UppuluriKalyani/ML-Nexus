# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
import logging
import math
import torch

logger = logging.getLogger(__name__)

@dataclass
class LabelSmoothedDualImitationCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )


@register_criterion(
    "ce_loss_ls", dataclass=LabelSmoothedDualImitationCriterionConfig
)
class CE_Loss_LS(LabelSmoothedDualImitationCriterion):
    def __init__(self, task, label_smoothing, report_accuracy=False):
        super().__init__(task, label_smoothing)
        self.report_accuracy = report_accuracy
        
    def _custom_loss(self, loss, name="loss", factor=1.0):
        loss = loss * factor
        return {"name": name, "loss": loss, "factor": factor}
  
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        net_input_concat = {}
        net_input_concat["sample"] = sample
        
        outputs, extra = model(**net_input_concat)
        
        losses, nll_loss = [], []
        
        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", self.label_smoothing),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )
            
        if self.report_accuracy:
            for obj in outputs:
                if outputs[obj].get("nll_loss", False):
                    n_correct, total = self.compute_accuracy(
                        outputs[obj]['out'], 
                        outputs[obj].get("tgt"),
                        outputs[obj].get("mask", None),
                        )
                    logging_output[f"n_correct_{obj}"] = utils.item(n_correct)
                    logging_output[f"total_{obj}"] = utils.item(total)
                    
        return loss, sample_size, logging_output
    
    def compute_accuracy(self, prediction, target, mask):      
        if mask is None:
            n_correct = torch.sum(
                prediction.argmax(-1).eq(target)
            )
            total = target.numel()
        else:
            n_correct = torch.sum(
                prediction.argmax(-1).masked_select(mask).eq(target.masked_select(mask))
            )
            total = torch.sum(mask)
        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
            
        for key in logging_outputs[0]:
            if "total" in key:
                n_c_name = "n_correct" + key[5:]
                total = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                if total > 0:
                    metrics.log_scalar_sum("_" + key, total)
                    n_correct = utils.item(
                        sum(log.get(n_c_name, 0) for log in logging_outputs)
                    )
                    metrics.log_scalar_sum("_" + n_c_name, n_correct)
                    
                    def cal_acc(meters, n_c_name="_" + n_c_name, key="_" + key):
                        return 100 * meters[n_c_name].sum / meters[key].sum
                    
                    metrics.log_derived(
                        f"accuracy{key[5:]}",
                        cal_acc
                        if total > 0
                        else float("nan"),
                    )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
