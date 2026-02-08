#!/usr/bin/env python3
"""
SST-2 Local Training Client Implementation
Standalone client for SST-2 sentiment analysis training
"""

import os
import csv
import logging
from typing import List, Dict, Tuple

from src.clients.base_local_client import BaseLocalClient

logger = logging.getLogger(__name__)

class SST2LocalClient(BaseLocalClient):
    """Local training client specialized for SST-2 sentiment analysis"""

    def __init__(self, config_path: str = None):
        super().__init__("sst2", config_path)

    def load_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Load SST-2 dataset"""
        try:
            # Try to load from datasets library first
            return self.load_dataset_from_library()
        except (ImportError, Exception) as e:
            # Fallback to local files or dummy data
            self.logger.warning(f"Could not load from datasets library: {e}")
            return self._load_local_sst2_data()

    def _convert_dataset_item(self, item: Dict) -> Dict:
        """Convert GLUE SST-2 dataset item to our format"""
        return {
            'text': item['sentence'],
            'label': item['label']
        }

    def _load_local_sst2_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load SST-2 data from local files or generate dummy data"""
        # Try to load from GLUE data first
        glue_data_path = "/home/pc/Documents/LAB/FedAvgLS/FedBERT-LoRA/glue_data/SST-2"

        if os.path.exists(glue_data_path):
            return self._load_glue_sst2_data(glue_data_path)
        else:
            # Fallback to generating dummy data for testing
            self.logger.warning("GLUE SST-2 data not found, using dummy data")
            return self._generate_dummy_sst2_data()

    def _load_glue_sst2_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load SST-2 data from GLUE dataset"""
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
                    if len(parts) >= 2:
                        # SST-2 format: sentence, label
                        sentence = parts[0]
                        label = int(parts[1])
                        train_data.append({
                            'text': sentence,
                            'label': label
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0]
                        label = int(parts[1])
                        val_data.append({
                            'text': sentence,
                            'label': label
                        })

        self.logger.info(f"Loaded {len(train_data)} SST-2 training samples")
        self.logger.info(f"Loaded {len(val_data)} SST-2 validation samples")

        return train_data, val_data

    def _generate_dummy_sst2_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy SST-2 data for testing"""
        # Positive sentiment examples
        positive_sentences = [
            "This movie is absolutely fantastic and wonderful",
            "I really enjoyed the beautiful scenery in this film",
            "The acting was superb and the story was engaging",
            "A masterpiece of modern cinema",
            "Brilliant performances by the entire cast",
            "The soundtrack perfectly complements the visuals",
            "An inspiring and uplifting story",
            "Visually stunning with great cinematography",
            "Excellent direction and screenplay",
            "A must-watch film for everyone"
        ]

        # Negative sentiment examples
        negative_sentences = [
            "This movie is terrible and boring",
            "The plot was confusing and poorly written",
            "Bad acting ruined the entire film",
            "A complete waste of time",
            "The worst movie I've seen this year",
            "Poor direction and terrible pacing",
            "Uninteresting characters and weak story",
            "The special effects were laughably bad",
            "Not worth the price of admission",
            "I regret watching this film"
        ]

        train_data = []
        val_data = []

        # Generate training data
        for i, sentence in enumerate(positive_sentences * 25):  # 250 positive samples
            train_data.append({
                'text': sentence,
                'label': 1  # Positive
            })

        for i, sentence in enumerate(negative_sentences * 25):  # 250 negative samples
            train_data.append({
                'text': sentence,
                'label': 0  # Negative
            })

        # Generate validation data
        for i, sentence in enumerate(positive_sentences[:20]):  # 20 positive samples
            val_data.append({
                'text': sentence,
                'label': 1  # Positive
            })

        for i, sentence in enumerate(negative_sentences[:20]):  # 20 negative samples
            val_data.append({
                'text': sentence,
                'label': 0  # Negative
            })

        self.logger.info(f"Generated {len(train_data)} dummy SST-2 training samples")
        self.logger.info(f"Generated {len(val_data)} dummy SST-2 validation samples")

        return train_data, val_data

    def _load_glue_sst2_data(self, data_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Load SST-2 data from GLUE dataset"""
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
                    if len(parts) >= 2:
                        # SST-2 format: sentence, label
                        sentence = parts[0]
                        label = int(parts[1])
                        train_data.append({
                            'text': sentence,
                            'label': label
                        })

        # Load validation data
        if os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                next(f)  # Skip header if exists
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0]
                        label = int(parts[1])
                        val_data.append({
                            'text': sentence,
                            'label': label
                        })

        self.logger.info(f"Loaded {len(train_data)} SST-2 training samples")
        self.logger.info(f"Loaded {len(val_data)} SST-2 validation samples")

        return train_data, val_data

    def _generate_dummy_sst2_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate dummy SST-2 data for testing"""
        # Positive sentiment examples
        positive_sentences = [
            "This movie is absolutely fantastic and wonderful",
            "I really enjoyed the beautiful scenery in this film",
            "The acting was superb and the story was engaging",
            "A masterpiece of modern cinema",
            "Brilliant performances by the entire cast",
            "The soundtrack perfectly complements the visuals",
            "An inspiring and uplifting story",
            "Visually stunning with great cinematography",
            "Excellent direction and screenplay",
            "A must-watch film for everyone"
        ]

        # Negative sentiment examples
        negative_sentences = [
            "This movie is terrible and boring",
            "The plot was confusing and poorly written",
            "Bad acting ruined the entire film",
            "A complete waste of time",
            "The worst movie I've seen this year",
            "Poor direction and terrible pacing",
            "Uninteresting characters and weak story",
            "The special effects were laughably bad",
            "Not worth the price of admission",
            "I regret watching this film"
        ]

        train_data = []
        val_data = []

        # Generate training data
        for i, sentence in enumerate(positive_sentences * 25):  # 250 positive samples
            train_data.append({
                'text': sentence,
                'label': 1  # Positive
            })

        for i, sentence in enumerate(negative_sentences * 25):  # 250 negative samples
            train_data.append({
                'text': sentence,
                'label': 0  # Negative
            })

        # Generate validation data
        for i, sentence in enumerate(positive_sentences[:20]):  # 20 positive samples
            val_data.append({
                'text': sentence,
                'label': 1  # Positive
            })

        for i, sentence in enumerate(negative_sentences[:20]):  # 20 negative samples
            val_data.append({
                'text': sentence,
                'label': 0  # Negative
            })

        self.logger.info(f"Generated {len(train_data)} dummy SST-2 training samples")
        self.logger.info(f"Generated {len(val_data)} dummy SST-2 validation samples")

        return train_data, val_data

def run_sst2_local_training(config_path: str = None):
    """Run standalone SST-2 local training"""
    client = SST2LocalClient(config_path)

    print("[TRAINING] Starting SST-2 Local Training")
    print("=" * 40)

    try:
        results = client.run_training()

        print("\n[SUCCESS] SST-2 Training Completed Successfully!")
        print(f"[STATS] Final Training Loss: {results['final_metrics'].get('loss', 0):.4f}")
        print(f"[STATS] Final Training Accuracy: {results['final_metrics'].get('accuracy', 0):.4f}")

        if results['val_metrics']:
            final_val = results['val_metrics'][-1]
            print(f"[STATS] Final Validation Loss: {final_val.get('loss', 0):.4f}")
            print(f"[STATS] Final Validation Accuracy: {final_val.get('accuracy', 0):.4f}")

        print(f"[INFO] Results saved to: {client.config['output_dir']}")

        return results

    except Exception as e:
        logger.error(f"SST-2 training failed: {str(e)}")
        print(f" Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SST-2 Local Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    run_sst2_local_training(args.config)
