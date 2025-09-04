"""
MachineTypes - A comprehensive module for machine learning data types and utilities.

This module provides a collection of data types, utilities, and common operations
used in machine learning and data science workflows.
"""

from .core import (
    # Data Types
    Tensor, Matrix, Vector, Scalar,
    Dataset, DataLoader, Batch,
    
    # Neural Network Components
    Layer, Dense, Conv2D, LSTM, GRU, Dropout, BatchNorm,
    Activation, ReLU, Sigmoid, Tanh, Softmax,
    
    # Optimizers
    Optimizer, SGD, Adam, RMSprop,
    
    # Loss Functions
    Loss, MSELoss, CrossEntropyLoss, BCELoss,
    
    # Metrics
    Metric, Accuracy, Precision, Recall, F1Score,
    
    # Utils
    one_hot, normalize, standardize, train_test_split
)

__version__ = '0.1.0'
