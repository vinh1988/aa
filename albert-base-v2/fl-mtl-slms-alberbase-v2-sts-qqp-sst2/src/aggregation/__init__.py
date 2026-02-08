#!/usr/bin/env python3
"""
Aggregation module for federated learning
"""

from .standard_aggregator import StandardAggregator
from .mtl_aggregator import MTLAggregator

__all__ = ['StandardAggregator', 'MTLAggregator']

