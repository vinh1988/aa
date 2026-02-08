"""
Local Training Clients Package
Standalone training clients for individual NLP tasks
"""

from .base_local_client import BaseLocalClient
from .sst2_local_client import SST2LocalClient, run_sst2_local_training
from .qqp_local_client import QQPLocalClient, run_qqp_local_training
from .stsb_local_client import STSBLlocalClient, run_stsb_local_training

__all__ = [
    'BaseLocalClient',
    'SST2LocalClient',
    'QQPLocalClient',
    'STSBLlocalClient',
    'run_sst2_local_training',
    'run_qqp_local_training',
    'run_stsb_local_training'
]
