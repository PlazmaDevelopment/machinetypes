"""
Core functionality for MachineTypes module.

This module provides essential data types and utilities for machine learning,
including tensors, neural network components, optimizers, and metrics.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

T = TypeVar('T')

# ======================
# Base Classes
# ======================

class Tensor(np.ndarray):
    """Base class for multi-dimensional arrays with automatic differentiation."""
    
    def __new__(cls, data: Union[np.ndarray, list, float, int], 
                requires_grad: bool = False) -> 'Tensor':
        obj = np.asarray(data).view(cls)
        obj.requires_grad = requires_grad
        obj.grad = None
        return obj
    
    def backward(self, grad: 'Tensor' = None) -> None:
        """Compute the gradient of the tensor."""
        if not self.requires_grad:
            return
            
        if grad is None:
            if self.size == 1:
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")
        
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self))
        
        self.grad += grad


class Matrix(Tensor):
    """2D matrix type with linear algebra operations."""
    def __new__(cls, data: Union[np.ndarray, list], **kwargs) -> 'Matrix':
        arr = np.array(data, dtype=np.float32)
        if len(arr.shape) != 2:
            raise ValueError("Matrix must be 2-dimensional")
        return super().__new__(cls, arr, **kwargs)


class Vector(Tensor):
    """1D vector type."""
    def __new__(cls, data: Union[np.ndarray, list], **kwargs) -> 'Vector':
        arr = np.array(data, dtype=np.float32).flatten()
        return super().__new__(cls, arr, **kwargs)


class Scalar(Tensor):
    """Scalar (0D tensor) type."""
    def __new__(cls, data: Union[float, int], **kwargs) -> 'Scalar':
        return super().__new__(cls, float(data), **kwargs)


@dataclass
class Dataset(Generic[T]):
    """Base dataset class for machine learning."""
    data: List[T]
    targets: List[Any] = None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[T, Any]:
        if self.targets is None:
            return self.data[idx]
        return self.data[idx], self.targets[idx]
    
    def split(self, test_size: float = 0.2, random_state: int = None) -> Tuple['Dataset', 'Dataset']:
        """Split dataset into training and testing sets."""
        if random_state is not None:
            random.seed(random_state)
            
        indices = list(range(len(self)))
        random.shuffle(indices)
        split_idx = int(len(indices) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_data = [self.data[i] for i in train_indices]
        train_targets = [self.targets[i] for i in train_indices] if self.targets is not None else None
        
        test_data = [self.data[i] for i in test_indices]
        test_targets = [self.targets[i] for i in test_indices] if self.targets is not None else None
        
        return Dataset(train_data, train_targets), Dataset(test_data, test_targets)


class DataLoader:
    """DataLoader for batching and shuffling data."""
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.current_idx = 0
        
        if self.shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self
    
    def __next__(self) -> Tuple[list, list]:
        if self.current_idx >= len(self.dataset):
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        
        if self.dataset.targets is not None:
            data = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            return data, targets
        return batch


# ======================
# Neural Network Components
# ======================

class Layer(ABC):
    """Base class for all neural network layers."""
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Backward pass."""
        pass
    
    def parameters(self) -> List[Tensor]:
        """Return list of trainable parameters."""
        return []


class Dense(Layer):
    """Fully connected layer."""
    
    def __init__(self, input_dim: int, output_dim: int):
        # He initialization
        scale = np.sqrt(2.0 / input_dim)
        self.weights = Tensor(np.random.randn(input_dim, output_dim) * scale, requires_grad=True)
        self.bias = Tensor(np.zeros(output_dim), requires_grad=True)
        self.input = None
    
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return x @ self.weights + self.bias
    
    def backward(self, grad: Tensor) -> Tensor:
        if self.weights.requires_grad:
            self.weights.grad = self.input.T @ grad
        if self.bias.requires_grad:
            self.bias.grad = grad.sum(axis=0)
        return grad @ self.weights.T
    
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]


class Activation(Layer):
    """Base class for activation functions."""
    def __init__(self):
        self.input = None
    
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        return self._forward(x)
    
    @abstractmethod
    def _forward(self, x: Tensor) -> Tensor:
        pass
    
    @abstractmethod
    def _backward(self, x: Tensor, grad: Tensor) -> Tensor:
        pass
    
    def backward(self, grad: Tensor) -> Tensor:
        return self._backward(self.input, grad)


class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    def _forward(self, x: Tensor) -> Tensor:
        return np.maximum(0, x)
    
    def _backward(self, x: Tensor, grad: Tensor) -> Tensor:
        return grad * (x > 0).astype(float)


class Sigmoid(Activation):
    """Sigmoid activation function."""
    def _forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + np.exp(-x))
    
    def _backward(self, x: Tensor, grad: Tensor) -> Tensor:
        sig = self._forward(x)
        return grad * sig * (1 - sig)


# ======================
# Optimizers
# ======================

class Optimizer(ABC):
    """Base class for optimizers."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
    
    @abstractmethod
    def step(self):
        """Perform a single optimization step."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
            param -= self.velocities[i]


class Adam(Optimizer):
    """Adam optimizer."""
    def __init__(self, parameters: List[Tensor], lr: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = [np.zeros_like(p) for p in parameters]  # First moment vector
        self.v = [np.zeros_like(p) for p in parameters]  # Second moment vector
    
    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ======================
# Loss Functions
# ======================

class Loss(ABC):
    """Base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Compute the loss."""
        pass
    
    @abstractmethod
    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the gradient of the loss."""
        pass


class MSELoss(Loss):
    """Mean Squared Error loss."""
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return 2 * (y_pred - y_true) / y_pred.size


class CrossEntropyLoss(Loss):
    """Cross Entropy loss for multi-class classification."""
    def forward(self, y_pred: Tensor, y_true: Tensor) -> float:
        # Numerically stable softmax
        exp = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Cross-entropy
        n = y_pred.shape[0]
        log_likelihood = -np.log(softmax[range(n), y_true] + 1e-8)
        return np.mean(log_likelihood)
    
    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Numerically stable softmax gradient
        exp = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        
        # Gradient of cross-entropy with softmax
        n = y_pred.shape[0]
        grad = softmax.copy()
        grad[range(n), y_true] -= 1
        return grad / n


# ======================
# Metrics
# ======================

class Metric(ABC):
    """Base class for metrics."""
    
    @abstractmethod
    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        """Update the metric with new predictions and true values."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute the metric value."""
        pass
    
    def reset(self) -> None:
        """Reset the metric state."""
        pass


class Accuracy(Metric):
    """Accuracy metric for classification tasks."""
    
    def __init__(self):
        self.correct = 0
        self.total = 0
    
    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        """Update the accuracy metric."""
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        self.correct += np.sum(y_pred == y_true)
        self.total += len(y_true)
    
    def compute(self) -> float:
        """Compute the accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self) -> None:
        """Reset the accuracy metric."""
        self.correct = 0
        self.total = 0


# ======================
# Utility Functions
# ======================

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class labels to one-hot encoded vectors."""
    return np.eye(num_classes)[y]


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    return (x - x_min) / (x_max - x_min + 1e-8)


def standardize(x: np.ndarray) -> np.ndarray:
    """Standardize data to have zero mean and unit variance."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / (std + 1e-8)


def train_test_split(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = None) -> tuple:
    """Split data into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    test_size = int(test_size * n_samples)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return x_train, x_test, y_train, y_test
