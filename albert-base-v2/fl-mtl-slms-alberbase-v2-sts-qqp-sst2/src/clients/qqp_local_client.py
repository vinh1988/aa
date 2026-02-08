#!/usr/bin/env python3
"""
QQP Local Training Client Implementation
Standalone client for QQP question pair matching training
"""

import os
import csv
import logging
from typing import List, Dict, Tuple

from src.clients.base_local_client import BaseLocalClient

logger = logging.getLogger(__name__)

class QQPLocalClient(BaseLocalClient):
    """Local training client specialized for QQP question pair matching"""

    def __init__(self, config_path: str = None):
        super().__init__("qqp", config_path)

    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Load QQP dataset"""
        try:
            # Try to load from datasets library first
            return self.load_dataset_from_library()
        except (ImportError, Exception) as e:
            # Fallback to local files or dummy data
            self.logger.warning(f"Could not load from datasets library: {e}")
            return self._load_local_qqp_data()

    def _convert_dataset_item(self, item: Dict) -> Dict:
        """Convert GLUE QQP dataset item to our format"""
        return {
            'text': f"{item['question1']} [SEP] {item['question2']}",
            'label': item['label']
        }

    def _load_local_qqp_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load QQP data from local files or generate dummy data"""
        # Try to load from GLUE data first
        glue_data_path = "/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/glue_data/QQP"

        if os.path.exists(glue_data_path):
            return self._load_glue_qqp_data(glue_data_path)
        else:
            # Fallback to generating dummy data for testing
            self.logger.warning("GLUE QQP data not found, using dummy data")
            return self._generate_dummy_qqp_data()

    def _load_glue_qqp_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load QQP data from GLUE dataset"""
        train_file = os.path.join(data_path, "train.tsv")
        dev_file = os.path.join(data_path, "dev.tsv")

        train_data = []
        val_data = []

        # Load training data
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        # QQP format: id, qid1, qid2, question1, question2, is_duplicate, label
                        question1 = parts[3]
                        question2 = parts[4]
                        is_duplicate = int(parts[5])

                        train_data.append({
                            'text': f"{question1} [SEP] {question2}",
                            'label': is_duplicate
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        question1 = parts[3]
                        question2 = parts[4]
                        is_duplicate = int(parts[5])

                        val_data.append({
                            'text': f"{question1} [SEP] {question2}",
                            'label': is_duplicate
                        })

        self.logger.info(f"Loaded {len(train_data)} QQP training samples")
        self.logger.info(f"Loaded {len(val_data)} QQP validation samples")

        return train_data, val_data

    def _generate_dummy_qqp_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy QQP data for testing"""
        # Duplicate question pairs
        duplicate_pairs = [
            ("What is the capital of France?", "What is the capital city of France?"),
            ("How to learn Python?", "What is the best way to learn Python programming?"),
            ("What time is it?", "Can you tell me what time it is now?"),
            ("Where is the nearest restaurant?", "Where can I find a restaurant nearby?"),
            ("How old are you?", "What is your current age?"),
            ("What is machine learning?", "Can you explain what machine learning is?"),
            ("How to cook pasta?", "What is the recipe for cooking pasta?"),
            ("What is the weather today?", "How is the weather looking today?"),
            ("Where do I buy groceries?", "Which store should I go to for groceries?"),
            ("How to fix a computer?", "What steps to take to repair a computer?")
        ]

        # Non-duplicate question pairs
        non_duplicate_pairs = [
            ("What is the capital of France?", "How to bake a cake?"),
            ("How to learn Python?", "What is the population of Tokyo?"),
            ("What time is it?", "How to change a tire?"),
            ("Where is the nearest restaurant?", "What is quantum physics?"),
            ("How old are you?", "How to grow tomatoes?"),
            ("What is machine learning?", "What is the price of Bitcoin?"),
            ("How to cook pasta?", "How to play guitar?"),
            ("What is the weather today?", "How to invest in stocks?"),
            ("Where do I buy groceries?", "What is artificial intelligence?"),
            ("How to fix a computer?", "How to speak Spanish?")
        ]

        train_data = []
        val_data = []

        # Generate training data
        for q1, q2 in duplicate_pairs * 15:  # 150 duplicate samples
            train_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 1  # Duplicate
            })

        for q1, q2 in non_duplicate_pairs * 15:  # 150 non-duplicate samples
            train_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 0  # Not duplicate
            })

        # Generate validation data
        for q1, q2 in duplicate_pairs[:10]:  # 10 duplicate samples
            val_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 1  # Duplicate
            })

        for q1, q2 in non_duplicate_pairs[:10]:  # 10 non-duplicate samples
            val_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 0  # Not duplicate
            })

        self.logger.info(f"Generated {len(train_data)} dummy QQP training samples")
        self.logger.info(f"Generated {len(val_data)} dummy QQP validation samples")

        return train_data, val_data

    def _load_glue_qqp_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load QQP data from GLUE dataset"""
        train_file = os.path.join(data_path, "train.tsv")
        dev_file = os.path.join(data_path, "dev.tsv")

        train_data = []
        val_data = []

        # Load training data
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        # QQP format: id, qid1, qid2, question1, question2, is_duplicate, label
                        question1 = parts[3]
                        question2 = parts[4]
                        is_duplicate = int(parts[5])

                        train_data.append({
                            'text': f"{question1} [SEP] {question2}",
                            'label': is_duplicate
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        question1 = parts[3]
                        question2 = parts[4]
                        is_duplicate = int(parts[5])

                        val_data.append({
                            'text': f"{question1} [SEP] {question2}",
                            'label': is_duplicate
                        })

        self.logger.info(f"Loaded {len(train_data)} QQP training samples")
        self.logger.info(f"Loaded {len(val_data)} QQP validation samples")

        return train_data, val_data

    def _generate_dummy_qqp_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy QQP data for testing"""
        # Duplicate question pairs
        duplicate_pairs = [
            ("What is the capital of France?", "What is the capital city of France?"),
            ("How to learn Python?", "What is the best way to learn Python programming?"),
            ("What time is it?", "Can you tell me what time it is now?"),
            ("Where is the nearest restaurant?", "Where can I find a restaurant nearby?"),
            ("How old are you?", "What is your current age?"),
            ("What is machine learning?", "Can you explain what machine learning is?"),
            ("How to cook pasta?", "What is the recipe for cooking pasta?"),
            ("What is the weather today?", "How is the weather looking today?"),
            ("Where do I buy groceries?", "Which store should I go to for groceries?"),
            ("How to fix a computer?", "What steps to take to repair a computer?")
        ]

        # Non-duplicate question pairs
        non_duplicate_pairs = [
            ("What is the capital of France?", "How to bake a cake?"),
            ("How to learn Python?", "What is the population of Tokyo?"),
            ("What time is it?", "How to change a tire?"),
            ("Where is the nearest restaurant?", "What is quantum physics?"),
            ("How old are you?", "How to grow tomatoes?"),
            ("What is machine learning?", "What is the price of Bitcoin?"),
            ("How to cook pasta?", "How to play guitar?"),
            ("What is the weather today?", "How to invest in stocks?"),
            ("Where do I buy groceries?", "What is artificial intelligence?"),
            ("How to fix a computer?", "How to speak Spanish?")
        ]

        train_data = []
        val_data = []

        # Generate training data
        for q1, q2 in duplicate_pairs * 15:  # 150 duplicate samples
            train_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 1  # Duplicate
            })

        for q1, q2 in non_duplicate_pairs * 15:  # 150 non-duplicate samples
            train_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 0  # Not duplicate
            })

        # Generate validation data
        for q1, q2 in duplicate_pairs[:10]:  # 10 duplicate samples
            val_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 1  # Duplicate
            })

        for q1, q2 in non_duplicate_pairs[:10]:  # 10 non-duplicate samples
            val_data.append({
                'text': f"{q1} [SEP] {q2}",
                'label': 0  # Not duplicate
            })

        self.logger.info(f"Generated {len(train_data)} dummy QQP training samples")
        self.logger.info(f"Generated {len(val_data)} dummy QQP validation samples")

        return train_data, val_data

def run_qqp_local_training(config_path: str = None):
    """Run standalone QQP local training"""
    client = QQPLocalClient(config_path)

    print("[TRAINING] Starting QQP Local Training")
    print("=" * 40)

    try:
        results = client.run_training()

        print("\n[SUCCESS] QQP Training Completed Successfully!")
        print(f"[STATS] Final Training Loss: {results['final_metrics'].get('loss', 0):.4f}")
        print(f"[STATS] Final Training Accuracy: {results['final_metrics'].get('accuracy', 0):.4f}")

        if results['val_metrics']:
            final_val = results['val_metrics'][-1]
            print(f"[STATS] Final Validation Loss: {final_val.get('loss', 0):.4f}")
            print(f"[STATS] Final Validation Accuracy: {final_val.get('accuracy', 0):.4f}")

        print(f"[ERROR] Training failed: {str(e)}")

        print(f"[INFO] Results saved to: {client.config['output_dir']}")

        return results

    except Exception as e:
        logger.error(f"QQP training failed: {str(e)}")
        print(f"[ERROR] Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run QQP Local Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    run_qqp_local_training(args.config)
