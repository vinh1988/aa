#!/usr/bin/env python3
"""
STSB Local Training Client Implementation
Standalone client for STSB semantic similarity training
"""

import os
import csv
import logging
from typing import List, Dict, Tuple

from src.clients.base_local_client import BaseLocalClient

logger = logging.getLogger(__name__)

class STSBLlocalClient(BaseLocalClient):
    """Local training client specialized for STSB semantic similarity regression"""

    def __init__(self, config_path: str = None):
        super().__init__("stsb", config_path)

    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Load STSB dataset"""
        try:
            # Try to load from datasets library first
            return self.load_dataset_from_library()
        except (ImportError, Exception) as e:
            # Fallback to local files or dummy data
            self.logger.warning(f"Could not load from datasets library: {e}")
            return self._load_local_stsb_data()

    def _convert_dataset_item(self, item: Dict) -> Dict:
        """Convert GLUE STS-B dataset item to our format"""
        return {
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'score': item['label']
        }

    def _load_local_stsb_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load STS-B data from local files or generate dummy data"""
        # Try to load from GLUE data first
        glue_data_path = "/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/glue_data/STS-B"

        if os.path.exists(glue_data_path):
            return self._load_glue_stsb_data(glue_data_path)
        else:
            # Fallback to generating dummy data for testing
            self.logger.warning("GLUE STS-B data not found, using dummy data")
            return self._generate_dummy_stsb_data()

    def _load_glue_stsb_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load STS-B data from GLUE dataset"""
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
                    if len(parts) >= 7:
                        # STS-B format: genre, filename, year, score, sentence1, sentence2
                        score = float(parts[4])
                        sentence1 = parts[5]
                        sentence2 = parts[6]

                        train_data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': score
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        score = float(parts[4])
                        sentence1 = parts[5]
                        sentence2 = parts[6]

                        val_data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': score
                        })

        self.logger.info(f"Loaded {len(train_data)} STS-B training samples")
        self.logger.info(f"Loaded {len(val_data)} STS-B validation samples")

        return train_data, val_data

    def _generate_dummy_stsb_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy STS-B data for testing"""
        # Similar sentence pairs with different similarity scores
        sentence_pairs = [
            # Very similar pairs (high scores)
            ("The cat sits on the mat", "A cat rests on a mat", 4.5),
            ("Machine learning is powerful", "AI has great capabilities", 4.2),
            ("The weather is nice today", "It's a beautiful day outside", 4.0),
            ("I love programming", "Software development is my passion", 4.3),
            ("The quick brown fox", "A fast auburn animal", 3.8),

            # Moderately similar pairs (medium scores)
            ("Python is a programming language", "Java is also used for coding", 2.5),
            ("The movie was entertaining", "I enjoyed watching the film", 3.0),
            ("She runs every morning", "He jogs daily in the park", 2.8),
            ("Mathematics is complex", "Physics requires calculations", 2.2),
            ("Cooking requires ingredients", "Baking needs flour and sugar", 2.7),

            # Less similar pairs (low scores)
            ("The stock market is volatile", "Ocean waves are unpredictable", 1.2),
            ("Quantum physics is fascinating", "Pizza toppings are delicious", 0.8),
            ("Climate change is concerning", "Smartphone batteries drain quickly", 1.0),
            ("Space exploration is expensive", "Coffee tastes bitter without sugar", 0.5),
            ("Renewable energy is important", "Video games are entertaining", 1.5)
        ]

        train_data = []
        val_data = []

        # Generate training data (more samples)
        for sent1, sent2, base_score in sentence_pairs * 20:  # 300 samples
            # Add some variation to scores
            import random
            score_variation = random.uniform(-0.3, 0.3)
            score = max(0, min(5, base_score + score_variation))

            train_data.append({
                'sentence1': sent1,
                'sentence2': sent2,
                'score': score
            })

        # Generate validation data
        for sent1, sent2, base_score in sentence_pairs[:10]:  # 10 samples
            # Add some variation to scores
            import random
            score_variation = random.uniform(-0.3, 0.3)
            score = max(0, min(5, base_score + score_variation))

            val_data.append({
                'sentence1': sent1,
                'sentence2': sent2,
                'score': score
            })

        self.logger.info(f"Generated {len(train_data)} dummy STS-B training samples")
        self.logger.info(f"Generated {len(val_data)} dummy STS-B validation samples")

        return train_data, val_data

    def _load_glue_stsb_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load STS-B data from GLUE dataset"""
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
                    if len(parts) >= 7:
                        # STS-B format: genre, filename, year, score, sentence1, sentence2
                        score = float(parts[4])
                        sentence1 = parts[5]
                        sentence2 = parts[6]

                        train_data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': score
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        score = float(parts[4])
                        sentence1 = parts[5]
                        sentence2 = parts[6]

                        val_data.append({
                            'sentence1': sentence1,
                            'sentence2': sentence2,
                            'score': score
                        })

        self.logger.info(f"Loaded {len(train_data)} STS-B training samples")
        self.logger.info(f"Loaded {len(val_data)} STS-B validation samples")

        return train_data, val_data

    def _generate_dummy_stsb_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy STS-B data for testing"""
        # Similar sentence pairs with different similarity scores
        sentence_pairs = [
            # Very similar pairs (high scores)
            ("The cat sits on the mat", "A cat rests on a mat", 4.5),
            ("Machine learning is powerful", "AI has great capabilities", 4.2),
            ("The weather is nice today", "It's a beautiful day outside", 4.0),
            ("I love programming", "Software development is my passion", 4.3),
            ("The quick brown fox", "A fast auburn animal", 3.8),

            # Moderately similar pairs (medium scores)
            ("Python is a programming language", "Java is also used for coding", 2.5),
            ("The movie was entertaining", "I enjoyed watching the film", 3.0),
            ("She runs every morning", "He jogs daily in the park", 2.8),
            ("Mathematics is complex", "Physics requires calculations", 2.2),
            ("Cooking requires ingredients", "Baking needs flour and sugar", 2.7),

            # Less similar pairs (low scores)
            ("The stock market is volatile", "Ocean waves are unpredictable", 1.2),
            ("Quantum physics is fascinating", "Pizza toppings are delicious", 0.8),
            ("Climate change is concerning", "Smartphone batteries drain quickly", 1.0),
            ("Space exploration is expensive", "Coffee tastes bitter without sugar", 0.5),
            ("Renewable energy is important", "Video games are entertaining", 1.5)
        ]

        train_data = []
        val_data = []

        # Generate training data (more samples)
        for sent1, sent2, base_score in sentence_pairs * 20:  # 300 samples
            # Add some variation to scores
            import random
            score_variation = random.uniform(-0.3, 0.3)
            score = max(0, min(5, base_score + score_variation))

            train_data.append({
                'sentence1': sent1,
                'sentence2': sent2,
                'score': score
            })

        # Generate validation data
        for sent1, sent2, base_score in sentence_pairs[:10]:  # 10 samples
            # Add some variation to scores
            import random
            score_variation = random.uniform(-0.3, 0.3)
            score = max(0, min(5, base_score + score_variation))

            val_data.append({
                'sentence1': sent1,
                'sentence2': sent2,
                'score': score
            })

        self.logger.info(f"Generated {len(train_data)} dummy STS-B training samples")
        self.logger.info(f"Generated {len(val_data)} dummy STS-B validation samples")

        return train_data, val_data

def run_stsb_local_training(config_path: str = None):
    """Run standalone STSB local training"""
    client = STSBLlocalClient(config_path)

    print("[STATS] Starting STSB Local Training")
    print("=" * 40)

    try:
        results = client.run_training()

        print("\n[SUCCESS] STSB Training Completed Successfully!")
        print("=" * 40)

        final_metrics = results['final_metrics']
        print(f"[STATS] Final Training Loss: {final_metrics.get('loss', 0):.4f}")
        print(f"[STATS] Final Training MAE: {final_metrics.get('mae', 0):.4f}")
        print(f"[STATS] Final Training Correlation: {final_metrics.get('correlation', 0):.4f}")

        if results['val_metrics']:
            final_val = results['val_metrics'][-1]
            print(f"[STATS] Final Validation Loss: {final_val.get('loss', 0):.4f}")
            print(f"[STATS] Final Validation MAE: {final_val.get('mae', 0):.4f}")
            print(f"[STATS] Final Validation Correlation: {final_val.get('correlation', 0):.4f}")

        print(f"[INFO] Results saved to: {client.config['output_dir']}")

        return results

    except Exception as e:
        logger.error(f"STS-B training failed: {str(e)}")
        print(f"[ERROR] Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run STSB Local Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    run_stsb_local_training(args.config)
