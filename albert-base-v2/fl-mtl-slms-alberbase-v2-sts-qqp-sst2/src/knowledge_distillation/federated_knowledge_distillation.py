#!/usr/bin/env python3
"""
Knowledge Distillation Implementation
Bidirectional KD for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class BidirectionalKDManager:
    """Manages bidirectional knowledge distillation"""

    def __init__(self, teacher_model, student_model, temperature: float = 3.0, alpha: float = 0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_history = []

    def teacher_to_student_kd_loss(self, student_logits, teacher_logits, labels, task_type="classification"):
        """Traditional KD: Teacher teaches student"""
        if task_type == "regression":
            # For regression tasks, use MSE loss for both KD and hard loss
            # Ensure tensors have compatible shapes
            if student_logits.shape != teacher_logits.shape:
                # Squeeze both tensors to ensure compatibility
                student_logits = student_logits.squeeze()
                teacher_logits = teacher_logits.squeeze()

            kd_loss = F.mse_loss(student_logits, teacher_logits)
            
            if labels is not None:
                hard_loss = F.mse_loss(student_logits.squeeze(), labels.float().squeeze())
                total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
            else:
                total_loss = kd_loss
        else:
            # For classification tasks, use KL divergence
            # Soft targets from teacher
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)

            # KL divergence loss for soft targets
            kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

            # Hard loss from ground truth labels
            if labels is not None:
                hard_loss = F.cross_entropy(student_logits, labels)
                # Combined loss
                total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
            else:
                total_loss = kd_loss

        return total_loss

    def student_to_teacher_kd_loss(self, student_logits, teacher_logits):
        """Reverse KD: Student teaches teacher"""
        # Ensure both tensors have the same shape for MSE loss
        if student_logits.shape != teacher_logits.shape:
            # If shapes don't match, squeeze or reshape to make them compatible
            student_logits = student_logits.squeeze()
            teacher_logits = teacher_logits.squeeze()

            # If still different shapes, take the minimum shape or broadcast
            if student_logits.shape != teacher_logits.shape:
                # Use the smaller dimension or broadcast to match
                min_shape = min(student_logits.shape, teacher_logits.shape)
                student_logits = student_logits.view(min_shape) if len(student_logits.shape) > len(min_shape) else student_logits
                teacher_logits = teacher_logits.view(min_shape) if len(teacher_logits.shape) > len(min_shape) else teacher_logits

        return F.mse_loss(teacher_logits, student_logits)

    def bidirectional_kd_loss(self, student_logits, teacher_logits, labels, reverse_weight: float = 0.1):
        """Combined bidirectional KD loss"""
        # Forward KD (teacher → student)
        forward_loss = self.teacher_to_student_kd_loss(student_logits, teacher_logits, labels)

        # Reverse KD (student → teacher)
        reverse_loss = self.student_to_teacher_kd_loss(student_logits, teacher_logits)

        # Combined loss with weighting
        total_loss = forward_loss + reverse_weight * reverse_loss

        # Record distillation event
        self.distillation_history.append({
            'timestamp': torch.tensor([0.0]),
            'forward_loss': forward_loss.item(),
            'reverse_loss': reverse_loss.item(),
            'total_loss': total_loss.item(),
            'temperature': self.temperature,
            'alpha': self.alpha
        })

        return total_loss

    def get_distillation_summary(self) -> Dict:
        """Get summary of distillation history"""
        if not self.distillation_history:
            return {}

        losses = [event['total_loss'] for event in self.distillation_history]
        return {
            'total_distillation_events': len(self.distillation_history),
            'average_total_loss': sum(losses) / len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'temperature_used': self.temperature,
            'alpha_used': self.alpha
        }

class LocalKDEngine:
    """Client-side KD engine for local training"""

    def __init__(self, student_model, tasks: List[str], config):
        self.student_model = student_model
        self.tasks = tasks
        self.config = config
        self.teacher_knowledge_cache = {}

    def update_teacher_knowledge(self, teacher_knowledge: Dict[str, torch.Tensor]):
        """Update cached teacher knowledge for KD"""
        self.teacher_knowledge_cache.update(teacher_knowledge)

    def calculate_kd_loss(self, student_logits: torch.Tensor, task_name: str, labels: torch.Tensor = None, current_round: int = 0) -> torch.Tensor:
        """Calculate KD loss for a specific task
        
        IMPROVED: Now supports disabling KD for initial rounds to establish baseline learning
        """
        # NEW: Check if KD should be used based on configuration
        use_kd = getattr(self.config, 'use_knowledge_distillation', False)
        kd_start_round = getattr(self.config, 'kd_start_round', 5)
        
        # Use simple loss if:
        # 1. KD is disabled in config, OR
        # 2. Current round is before kd_start_round, OR
        # 3. Teacher knowledge not available
        if not use_kd or current_round < kd_start_round or task_name not in self.teacher_knowledge_cache:
            # IMPROVED: Use only hard loss for better initial learning
            if labels is not None:
                # Use appropriate loss function based on task type
                if task_name == 'stsb':  # Regression task
                    return F.mse_loss(student_logits.squeeze(), labels.float().squeeze())
                else:  # Classification tasks
                    return F.cross_entropy(student_logits, labels)
            return torch.tensor(0.0, device=student_logits.device)

        # Use KD only after baseline learning is established
        teacher_logits = self.teacher_knowledge_cache[task_name]

        # Create KD manager for this calculation
        kd_manager = BidirectionalKDManager(
            None, None,  # We don't need full models for loss calculation
            temperature=self.config.kd_temperature,
            alpha=self.config.kd_alpha
        )

        # Determine task type
        task_type = "regression" if task_name == "stsb" else "classification"
        return kd_manager.teacher_to_student_kd_loss(student_logits, teacher_logits, labels, task_type)

    def prepare_student_knowledge_for_teacher(self, task_data: Dict) -> Dict[str, torch.Tensor]:
        """Prepare student knowledge to send back to teacher (with memory-efficient batching)"""
        student_knowledge = {}
        
        # Limit samples for knowledge distillation to prevent memory issues
        max_samples_for_kd = 1000  # Increased from 100 to 1000 for better reverse KD
        batch_size = 8  # Process in small batches

        for task_name in self.tasks:
            if task_name in task_data:
                # Get the data for this task
                task_data_item = task_data[task_name]
                
                # Check if we have tokenized data or need to tokenize
                if 'input_ids' in task_data_item and 'attention_mask' in task_data_item:
                    # Data is already tokenized
                    input_ids = task_data_item['input_ids']
                    attention_mask = task_data_item['attention_mask']
                else:
                    # Data needs tokenization - this shouldn't happen with our current setup
                    # but we'll handle it gracefully
                    continue
                
                # Limit to max_samples_for_kd to prevent memory issues
                if input_ids.size(0) > max_samples_for_kd:
                    logger.info(f"Limiting student knowledge samples for {task_name} from {input_ids.size(0)} to {max_samples_for_kd}")
                    input_ids = input_ids[:max_samples_for_kd]
                    attention_mask = attention_mask[:max_samples_for_kd]
                
                # Ensure tensors are on the correct device
                device = next(self.student_model.parameters()).device
                
                # Process in batches to avoid memory issues
                all_logits = []
                num_samples = input_ids.size(0)
                
                with torch.no_grad():
                    for i in range(0, num_samples, batch_size):
                        # Get batch
                        batch_input_ids = input_ids[i:i+batch_size].to(device)
                        batch_attention_mask = attention_mask[i:i+batch_size].to(device)
                        
                        # Generate student predictions for this batch
                        batch_logits = self.student_model(
                            batch_input_ids,
                            batch_attention_mask,
                            task_name
                        )
                        
                        # Move to CPU to save GPU memory
                        all_logits.append(batch_logits.cpu())
                        
                        # Clean up batch tensors
                        del batch_input_ids, batch_attention_mask, batch_logits
                        
                        # Clear GPU cache periodically
                        if torch.cuda.is_available() and i % (batch_size * 5) == 0:
                            torch.cuda.empty_cache()
                    
                    # Concatenate all batches and keep on CPU initially
                    student_knowledge[task_name] = torch.cat(all_logits, dim=0)
                    
                    # Clean up
                    del all_logits
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Prepared student knowledge for {task_name}: {student_knowledge[task_name].shape}")

        return student_knowledge

class GlobalKDManager:
    """Server-side KD manager for global knowledge management"""

    def __init__(self, teacher_model, config):
        self.teacher_model = teacher_model
        self.config = config
        self.global_knowledge_base = {}
        self.kd_optimizer = None

        # Only create optimizer if teacher model is available
        if teacher_model is not None:
            self.kd_optimizer = torch.optim.AdamW(self.teacher_model.parameters(), lr=1e-4)

    def generate_teacher_knowledge(self, sample_inputs: Dict[str, Dict] = None) -> Dict[str, torch.Tensor]:
        """Generate teacher knowledge (soft labels) for all tasks"""
        teacher_knowledge = {}

        if sample_inputs:
            # Generate knowledge using sample inputs
            if self.teacher_model is not None:
                with torch.no_grad():
                    for task_name, inputs in sample_inputs.items():
                        teacher_logits = self.teacher_model(
                            inputs['input_ids'],
                            inputs['attention_mask']
                        )
                        teacher_knowledge[task_name] = teacher_logits
            else:
                # Generate placeholder knowledge if no teacher model
                for task_name, inputs in sample_inputs.items():
                    batch_size = inputs['input_ids'].shape[0]
                    if task_name in ['sst2', 'qqp']:
                        # Binary classification: (batch_size, 2)
                        teacher_knowledge[task_name] = torch.randn(batch_size, 2, dtype=torch.float32)
                    else:
                        # Regression: (batch_size, 1)
                        teacher_knowledge[task_name] = torch.randn(batch_size, 1, dtype=torch.float32)
        else:
            # Generate placeholder knowledge (can be enhanced with actual data)
            for task in ['sst2', 'qqp', 'stsb']:
                # Create dummy knowledge for demonstration
                if task in ['sst2', 'qqp']:
                    teacher_knowledge[task] = torch.randn(1, 2)  # Binary classification
                else:
                    teacher_knowledge[task] = torch.randn(1, 1)  # Regression

        # Cache for future use
        self.global_knowledge_base.update(teacher_knowledge)

        return teacher_knowledge

    def update_teacher_from_students(self, student_knowledge_updates: List[Dict]) -> Dict:
        """Update teacher model using student knowledge (reverse KD)"""
        if not student_knowledge_updates:
            return {"updated": False, "reason": "No student knowledge provided"}

        total_loss = 0.0
        num_updates = 0

        for update in student_knowledge_updates:
            student_knowledge = update.get('student_knowledge', {})

            for task_name, student_logits in student_knowledge.items():
                if task_name in self.global_knowledge_base:
                    # Teacher learns from student's predictions only if teacher model exists
                    if self.teacher_model is not None:
                        teacher_logits = self.teacher_model(student_logits)

                        # Reverse KD loss - ensure tensors have compatible shapes
                        if teacher_logits.shape != student_logits.shape:
                            # Squeeze both tensors to ensure compatibility
                            teacher_logits = teacher_logits.squeeze()
                            student_logits = student_logits.squeeze()

                        reverse_loss = F.mse_loss(teacher_logits, student_logits)

                        total_loss += reverse_loss.item()
                        num_updates += 1

        if num_updates > 0:
            # Update teacher model only if optimizer exists
            if self.kd_optimizer is not None:
                avg_loss = total_loss / num_updates
                self.kd_optimizer.step()
                self.kd_optimizer.zero_grad()

            return {
                "updated": True,
                "avg_reverse_loss": avg_loss,
                "num_updates": num_updates,
                "tasks_updated": list(self.global_knowledge_base.keys())
            }

        return {"updated": False, "reason": "No valid updates"}

    def get_teacher_knowledge_summary(self) -> Dict:
        """Get summary of teacher knowledge state"""
        return {
            'cached_tasks': list(self.global_knowledge_base.keys()),
            'knowledge_temperature': self.config.kd_temperature,
            'knowledge_alpha': self.config.kd_alpha,
            'teacher_model_frozen': all(not p.requires_grad for p in self.teacher_model.parameters())
        }
