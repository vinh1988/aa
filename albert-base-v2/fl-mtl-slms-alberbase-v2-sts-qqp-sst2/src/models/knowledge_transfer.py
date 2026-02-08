from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressiveTransferConfig:
    start_round: int = 10
    end_round: int = 40
    max_weight: float = 0.5
    min_weight: float = 0.0

@dataclass
class DynamicAlignmentConfig:
    alignment_loss_weight: float = 0.1
    temperature: float = 2.0


class AdaptiveKnowledgeTransfer:
    def __init__(self, progressive_config: ProgressiveTransferConfig, alignment_config: DynamicAlignmentConfig):
        self.progressive_config = progressive_config
        self.alignment_config = alignment_config
        logger.info("AdaptiveKnowledgeTransfer initialized")

    def get_transfer_weight(self, current_round: int) -> float:
        if current_round < self.progressive_config.start_round:
            return self.progressive_config.min_weight
        elif current_round >= self.progressive_config.end_round:
            return self.progressive_config.max_weight
        else:
            # Linear interpolation for progressive transfer
            progress = (current_round - self.progressive_config.start_round) / \
                       (self.progressive_config.end_round - self.progressive_config.start_round)
            return self.progressive_config.min_weight + progress * \
                   (self.progressive_config.max_weight - self.progressive_config.min_weight)

    def compute_transfer_loss(self, 
                              current_round: int, 
                              client_outputs: Dict[str, Any], 
                              server_outputs: Dict[str, Any], 
                              attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        transfer_weight = self.get_transfer_weight(current_round)
        if transfer_weight == 0:
            return {"total_transfer_loss": torch.tensor(0.0)}

        # Example: Simple distillation loss on logits
        student_logits = client_outputs["logits"]
        teacher_logits = server_outputs["logits"].to(student_logits.device)

        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_logits / self.alignment_config.temperature, dim=-1),
            nn.functional.softmax(teacher_logits / self.alignment_config.temperature, dim=-1)
        ) * (self.alignment_config.temperature ** 2)
        
        total_transfer_loss = distillation_loss * self.alignment_config.alignment_loss_weight

        return {"total_transfer_loss": total_transfer_loss, "distillation_loss": distillation_loss}


class FederatedDistillationLoss:
    def __init__(self):
        logger.info("FederatedDistillationLoss initialized")

    def compute_loss(self, task_loss: torch.Tensor, transfer_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        total_transfer_loss = transfer_losses.get("total_transfer_loss", torch.tensor(0.0))
        total_loss = task_loss + total_transfer_loss
        return {"total_loss": total_loss, "task_loss": task_loss, "total_transfer_loss": total_transfer_loss}
