#!/usr/bin/env python3
"""
Federated Learning Main Entry Point
Orchestrates server and client modes with modular architecture
(Standard FL - No LoRA, No KD)
"""

import asyncio
import argparse
import sys
from typing import List

# Import from the modular structure
from federated_config import FederatedConfig, load_config
from src.core.federated_server import run_server
from src.core.federated_client import run_client

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Federated Learning System (Standard FL)")

    # Mode selection
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                       help="Run mode: server or client")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--config-file", type=str, default="federated_config.yaml",
                       help="Configuration file name")

    # Common arguments
    parser.add_argument("--port", type=int, default=8771, help="Server port")
    parser.add_argument("--rounds", type=int, default=2, help="Number of training rounds")
    parser.add_argument("--samples", type=int, default=100, help="Samples per client")

    # Client-specific arguments
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--tasks", nargs='+', choices=["sst2", "qqp", "stsb"],
                       help="Task names for client (space-separated)")

    # Advanced arguments
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")

    return parser

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args)

        # Override config with command line arguments
        if args.port != 8771:
            config.port = args.port
        if args.rounds != 2:
            config.num_rounds = args.rounds
        if args.samples != 100:
            config.samples_per_client = args.samples
        if args.log_level != "INFO":
            config.log_level = args.log_level

        print("[CONFIG] Federated Learning Configuration (MTL - Server-Side)")
        print("=" * 60)
        print(f"[MODEL] Model: {config.server_model} (server), {config.client_model} (client)")
        print(f"[SYNC] Sync: {'Enabled' if config.enable_synchronization else 'Disabled'} ({config.sync_frequency})")
        print(f"[TRAINING] Training: {config.num_rounds} rounds, {config.local_epochs} epochs, batch_size={config.batch_size}")
        print(f"[DATA] Task-Specific Samples:")
        for task, task_config in config.task_configs.items():
            train = task_config.get('train_samples', 'N/A')
            val = task_config.get('val_samples', 'N/A')
            print(f"       {task.upper()}: {train} train, {val} val")
        print(f"[NETWORK] Communication: Port {config.port}, timeout={config.timeout}s")
        print(f"[OUTPUT] Output: Results in '{config.results_dir}', log_level={config.log_level}")
        print("=" * 60)
        # Run federated system
        if args.mode == "server":
            print("[SERVER] Starting Federated Server...")
            asyncio.run(run_server(config))
        else:
            print(f"[CLIENT] Starting Federated Client: {args.client_id}...")
            if not args.client_id:
                print("[ERROR] Error: Client ID is required for client mode")
                sys.exit(1)
            if not args.tasks:
                print("[ERROR] Error: Tasks are required for client mode")
                sys.exit(1)
            client_config = config
            run_client(args.client_id, args.tasks, client_config)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
