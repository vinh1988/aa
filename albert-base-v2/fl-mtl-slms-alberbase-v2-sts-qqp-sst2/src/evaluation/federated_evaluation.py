#!/usr/bin/env python3
"""
Federated Learning Evaluation Module
Comprehensive evaluation system for federated learning models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for federated learning"""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device() if device == "auto" else torch.device(device)

    def _get_device(self):
        """Get available device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_client_model(self, model, dataloader, task_type: str) -> Dict[str, float]:
        """Evaluate a client model on its validation data"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = model(**inputs, task_name=task_type)
                loss = F.cross_entropy(outputs, labels) if task_type in ['sst2', 'qqp'] else F.mse_loss(outputs.squeeze(), labels.float().squeeze())
                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                if task_type in ['sst2', 'qqp']:
                    predictions = torch.argmax(outputs, dim=1)
                else:  # regression
                    predictions = outputs.squeeze()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics based on task type
        if task_type in ['sst2', 'qqp']:
            return self._calculate_classification_metrics(all_labels, all_predictions, total_loss / num_batches)
        else:  # stsb regression
            return self._calculate_regression_metrics(all_labels, all_predictions, total_loss / num_batches)

    def _calculate_classification_metrics(self, labels: List, predictions: List, avg_loss: float) -> Dict[str, float]:
        """Calculate classification metrics"""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'validation_loss': float(avg_loss)
        }

    def _calculate_regression_metrics(self, labels: List, predictions: List, avg_loss: float) -> Dict[str, float]:
        """Calculate regression metrics"""
        labels = np.array(labels)
        predictions = np.array(predictions)

        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)

        # Pearson correlation coefficient
        if len(labels) > 1:
            pearson_corr = np.corrcoef(labels, predictions)[0, 1]
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        else:
            pearson_corr = 0.0

        # Spearman correlation coefficient
        if len(labels) > 1:
            spearman_corr, _ = spearmanr(labels, predictions)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        else:
            spearman_corr = 0.0

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'pearson_correlation': float(pearson_corr),
            'spearman_correlation': float(spearman_corr),
            'validation_loss': float(avg_loss)
        }

class GlobalModelEvaluator:
    """Evaluate global model across all tasks and clients"""

    def __init__(self, device: str = "auto"):
        self.device = self._get_device() if device == "auto" else torch.device(device)
        self.evaluation_history = []

    def _get_device(self):
        """Get available device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_global_model(self, global_model, client_validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate global model on aggregated validation data from all clients"""
        global_metrics = {}
        task_aggregated_metrics = {}

        for client_id, client_data in client_validation_data.items():
            for task_name, task_data in client_data.items():
                if task_name not in task_aggregated_metrics:
                    task_aggregated_metrics[task_name] = {
                        'predictions': [],
                        'labels': [],
                        'client_contributions': []
                    }

                # Evaluate this client's contribution to this task
                evaluator = ModelEvaluator(self.device)
                metrics = evaluator.evaluate_client_model(
                    global_model,
                    task_data['dataloader'],
                    task_name
                )

                # Aggregate predictions and labels for this task
                task_aggregated_metrics[task_name]['predictions'].extend(task_data['predictions'])
                task_aggregated_metrics[task_name]['labels'].extend(task_data['labels'])
                task_aggregated_metrics[task_name]['client_contributions'].append({
                    'client_id': client_id,
                    'metrics': metrics
                })

        # Calculate aggregated metrics per task
        for task_name, task_data in task_aggregated_metrics.items():
            if task_name in ['sst2', 'qqp']:
                # Classification metrics
                accuracy = accuracy_score(task_data['labels'], task_data['predictions'])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    task_data['labels'], task_data['predictions'], average='weighted'
                )

                global_metrics[task_name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'num_clients': len(task_data['client_contributions']),
                    'total_samples': len(task_data['labels'])
                }
            else:
                # Regression metrics
                mse = mean_squared_error(task_data['labels'], task_data['predictions'])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(task_data['labels'], task_data['predictions'])
                
                # Pearson correlation
                pearson_corr = np.corrcoef(task_data['labels'], task_data['predictions'])[0, 1]
                if np.isnan(pearson_corr):
                    pearson_corr = 0.0
                
                # Spearman correlation
                spearman_corr, _ = spearmanr(task_data['labels'], task_data['predictions'])
                if np.isnan(spearman_corr):
                    spearman_corr = 0.0

                global_metrics[task_name] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'pearson_correlation': float(pearson_corr),
                    'spearman_correlation': float(spearman_corr),
                    'num_clients': len(task_data['client_contributions']),
                    'total_samples': len(task_data['labels'])
                }

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(global_metrics)

        # Record evaluation
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'global_metrics': global_metrics,
            'overall_metrics': overall_metrics,
            'task_aggregated_metrics': task_aggregated_metrics
        }
        self.evaluation_history.append(evaluation_record)

        return {
            'global_metrics': global_metrics,
            'overall_metrics': overall_metrics,
            'task_aggregated_metrics': task_aggregated_metrics,
            'evaluation_timestamp': datetime.now().isoformat()
        }

    def _calculate_overall_metrics(self, task_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate overall metrics across all tasks"""
        total_samples = 0
        weighted_accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

        for task_name, metrics in task_metrics.items():
            samples = metrics['total_samples']
            total_samples += samples

            if task_name in ['sst2', 'qqp']:
                # Classification tasks
                weighted_accuracy += metrics['accuracy'] * samples
                macro_f1 += metrics['f1_score']  # For macro average
                weighted_f1 += metrics['f1_score'] * samples
            else:
                # For regression tasks, use negative MSE as "accuracy" proxy
                # This allows consistent overall metrics
                regression_score = 1.0 / (1.0 + metrics['mse'])  # Convert MSE to 0-1 scale
                weighted_accuracy += regression_score * samples
                macro_f1 += regression_score
                weighted_f1 += regression_score * samples

        if total_samples > 0:
            overall_accuracy = weighted_accuracy / total_samples
            macro_f1_avg = macro_f1 / len(task_metrics)
            weighted_f1_avg = weighted_f1 / total_samples
        else:
            overall_accuracy = 0.0
            macro_f1_avg = 0.0
            weighted_f1_avg = 0.0

        return {
            'overall_accuracy': overall_accuracy,
            'macro_f1_score': macro_f1_avg,
            'weighted_f1_score': weighted_f1_avg,
            'total_samples': total_samples,
            'num_tasks': len(task_metrics)
        }

class EvaluationReporter:
    """Generate comprehensive evaluation reports"""

    def __init__(self, results_dir: str = "federated_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], round_num: int) -> str:
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_report_round_{round_num}_{timestamp}.json"
        report_path = os.path.join(self.results_dir, report_filename)

        # Add metadata
        report = {
            'metadata': {
                'round_number': round_num,
                'evaluation_timestamp': evaluation_results['evaluation_timestamp'],
                'report_generated_at': datetime.now().isoformat(),
                'system_version': 'federated_learning_v1.0'
            },
            'evaluation_results': evaluation_results
        }

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate human-readable summary
        summary = self._generate_human_readable_summary(evaluation_results, round_num)
        summary_filename = f"evaluation_summary_round_{round_num}_{timestamp}.txt"
        summary_path = os.path.join(self.results_dir, summary_filename)

        with open(summary_path, 'w') as f:
            f.write(summary)

        logger.info(f"Evaluation report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")

        return report_path

    def _generate_human_readable_summary(self, results: Dict[str, Any], round_num: int) -> str:
        """Generate human-readable evaluation summary"""
        summary = []
        summary.append(" Federated Learning Evaluation Report")
        summary.append("=" * 50)
        summary.append(f"Round: {round_num}")
        summary.append(f"Evaluation Time: {results['evaluation_timestamp']}")
        summary.append("")

        # Overall metrics
        overall = results['overall_metrics']
        summary.append(" Overall Performance:")
        summary.append(f"  • Overall Accuracy: {overall['overall_accuracy']:.4f}")
        summary.append(f"  • Macro F1 Score: {overall['macro_f1_score']:.4f}")
        summary.append(f"  • Weighted F1 Score: {overall['weighted_f1_score']:.4f}")
        summary.append(f"  • Total Samples: {overall['total_samples']}")
        summary.append(f"  • Tasks Evaluated: {overall['num_tasks']}")
        summary.append("")

        # Task-specific metrics
        summary.append(" Task-Specific Performance:")
        for task_name, metrics in results['global_metrics'].items():
            summary.append(f"  • {task_name.upper()}:")
            if task_name in ['sst2', 'qqp']:
                summary.append(f"    - Accuracy: {metrics['accuracy']:.4f}")
                summary.append(f"    - F1 Score: {metrics['f1_score']:.4f}")
                summary.append(f"    - Precision: {metrics['precision']:.4f}")
                summary.append(f"    - Recall: {metrics['recall']:.4f}")
            else:
                summary.append(f"    - MSE: {metrics['mse']:.4f}")
                summary.append(f"    - RMSE: {metrics['rmse']:.4f}")
                summary.append(f"    - MAE: {metrics['mae']:.4f}")
                summary.append(f"    - Pearson Correlation: {metrics['pearson_correlation']:.4f}")
                summary.append(f"    - Spearman Correlation: {metrics['spearman_correlation']:.4f}")
            summary.append(f"    - Clients: {metrics['num_clients']}")
            summary.append(f"    - Samples: {metrics['total_samples']}")
            summary.append("")

        # Client contributions
        summary.append("👥 Client Contributions:")
        for task_name, task_data in results['task_aggregated_metrics'].items():
            summary.append(f"  • {task_name.upper()} Clients:")
            for contribution in task_data['client_contributions']:
                client_metrics = contribution['metrics']
                summary.append(f"    - {contribution['client_id']}:")
                if task_name in ['sst2', 'qqp']:
                    summary.append(f"      Accuracy: {client_metrics['accuracy']:.4f}")
                else:
                    summary.append(f"      MSE: {client_metrics['mse']:.4f}")
            summary.append("")

        summary.append("=" * 50)
        summary.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(summary)

def create_evaluation_dataloaders(client_validation_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create evaluation dataloaders from client validation data"""
    evaluation_dataloaders = {}

    for client_id, client_data in client_validation_data.items():
        evaluation_dataloaders[client_id] = {}
        for task_name, task_data in client_data.items():
            # Create a simple dataloader from the validation data
            from torch.utils.data import DataLoader, TensorDataset

            # Extract validation data
            val_texts = task_data.get('val_texts', [])
            val_labels = task_data.get('val_labels', [])

            if val_texts and val_labels:
                # Create dummy input_ids and attention_mask (simplified for demo)
                input_ids = torch.randint(0, 1000, (len(val_texts), 128))
                attention_mask = torch.ones(len(val_texts), 128)

                dataset = TensorDataset(input_ids, attention_mask, torch.tensor(val_labels))
                dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

                evaluation_dataloaders[client_id][task_name] = {
                    'dataloader': dataloader,
                    'predictions': [],  # Will be filled during evaluation
                    'labels': val_labels
                }

    return evaluation_dataloaders

class PerformanceTracker:
    """Track performance across training rounds"""

    def __init__(self, results_dir: str = "federated_results"):
        self.results_dir = results_dir
        self.performance_history = []
        os.makedirs(results_dir, exist_ok=True)

    def record_round_performance(self, round_num: int, client_metrics: Dict[str, Dict], global_metrics: Dict[str, Any]):
        """Record performance metrics for a training round"""
        record = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'client_metrics': client_metrics,
            'global_metrics': global_metrics,
            'performance_summary': self._calculate_performance_summary(client_metrics, global_metrics)
        }

        self.performance_history.append(record)

        # Save to file
        self._save_performance_record(record)

    def _calculate_performance_summary(self, client_metrics: Dict[str, Dict], global_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary performance metrics"""
        summary = {}

        # Client-level summary
        if client_metrics:
            accuracies = [metrics.get('accuracy', 0) for metrics in client_metrics.values()]
            summary['avg_client_accuracy'] = sum(accuracies) / len(accuracies)

        # Global-level summary
        if global_metrics and 'overall_metrics' in global_metrics:
            overall = global_metrics['overall_metrics']
            summary.update({
                'global_accuracy': overall.get('overall_accuracy', 0),
                'macro_f1': overall.get('macro_f1_score', 0),
                'weighted_f1': overall.get('weighted_f1_score', 0)
            })

        return summary

    def _save_performance_record(self, record: Dict[str, Any]):
        """Save performance record to file"""
        filename = f"performance_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(record, f, indent=2, default=str)

    def get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends across rounds"""
        trends = {
            'rounds': [],
            'global_accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'avg_client_accuracy': []
        }

        for record in self.performance_history:
            trends['rounds'].append(record['round'])
            summary = record['performance_summary']

            trends['global_accuracy'].append(summary.get('global_accuracy', 0))
            trends['macro_f1'].append(summary.get('macro_f1', 0))
            trends['weighted_f1'].append(summary.get('weighted_f1', 0))
            trends['avg_client_accuracy'].append(summary.get('avg_client_accuracy', 0))

        return trends

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return "No performance data available."

        summary = []
        summary.append("[STATS] Overall Performance:")
        summary.append("[CLIENTS] Client Contributions:")
        report = []
        report.append("[REPORT] Federated Learning Performance Report")
        report.append("[TRENDS] Performance trends:")
        report.append(f"Total Rounds: {len(trends['rounds'])}")
        report.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall trends
        report.append(f"  • Global Accuracy: {trends['global_accuracy'][-1]:.4f} (final)")
        report.append(f"  • Macro F1 Score: {trends['macro_f1'][-1]:.4f} (final)")
        report.append(f"  • Weighted F1 Score: {trends['weighted_f1'][-1]:.4f} (final)")
        report.append(f"  • Avg Client Accuracy: {trends['avg_client_accuracy'][-1]:.4f} (final)")
        report.append("")

        # Round-by-round details
        report.append("[DETAILS] Round-by-Round Performance:")
        for i, round_num in enumerate(trends['rounds']):
            report.append(f"  Round {round_num}:")
            report.append(f"    - Global Accuracy: {trends['global_accuracy'][i]:.4f}")
            report.append(f"    - Macro F1: {trends['macro_f1'][i]:.4f}")
            report.append(f"    - Weighted F1: {trends['weighted_f1'][i]:.4f}")
            report.append(f"    - Avg Client Accuracy: {trends['avg_client_accuracy'][i]:.4f}")
        report.append("")

        report.append("=" * 50)
        report.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Save report
        report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(self.results_dir, report_filename)

        with open(report_path, 'w') as f:
            f.write("\n".join(report))

        logger.info(f"Performance report saved to {report_path}")
        return "\n".join(report)

# Utility functions for evaluation
def evaluate_model_performance(model, test_dataloader, task_type: str) -> Dict[str, float]:
    """Quick evaluation utility function"""
    evaluator = ModelEvaluator()
    return evaluator.evaluate_client_model(model, test_dataloader, task_type)

def generate_evaluation_summary(evaluation_results: Dict[str, Any]) -> str:
    """Generate a quick evaluation summary"""
    reporter = EvaluationReporter()
    return reporter._generate_human_readable_summary(evaluation_results, 0)
