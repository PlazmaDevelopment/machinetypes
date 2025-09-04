# MachineTypes

A comprehensive Python module for machine learning data types and utilities. This module provides a collection of essential data structures, neural network components, optimizers, loss functions, and metrics to simplify the development of machine learning models.

## Features

- **Tensor Operations**: Multi-dimensional arrays with automatic differentiation
- **Neural Network Components**: Layers, activations, and model building blocks
- **Optimization**: Various optimizers including SGD, Adam, and RMSprop
- **Loss Functions**: Common loss functions for different tasks
- **Metrics**: Evaluation metrics for model performance
- **Data Handling**: Dataset and DataLoader for efficient data loading
- **Utilities**: Common data preprocessing and manipulation functions

## Installation

```bash
pip install machinetypes
```

## Quick Start

```python
import numpy as np
from machinetypes import Tensor, Dense, ReLU, MSELoss, SGD, Dataset, DataLoader

# Create a simple neural network
class SimpleNN:
    def __init__(self):
        self.layer1 = Dense(2, 4)
        self.activation1 = ReLU()
        self.layer2 = Dense(4, 1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        return self.layer2(x)

# Create sample data
X = Tensor(np.random.randn(100, 2))
y = Tensor(np.random.randn(100, 1))
dataset = Dataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# Initialize model, loss, and optimizer
model = SimpleNN()
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model.forward(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
```

## Core Components

### Tensors

```python
from machinetypes import Tensor, Matrix, Vector, Scalar

# Create tensors
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Operations
c = a + b  # Element-wise addition
d = a * b  # Element-wise multiplication
e = a @ b  # Dot product

# Automatic differentiation
e.backward()
print(a.grad)  # Gradient of e with respect to a
```

### Neural Network Layers

```python
from machinetypes import Dense, ReLU, Sigmoid

# Create layers
layer1 = Dense(input_dim=10, output_dim=20)
activation = ReLU()
layer2 = Dense(input_dim=20, output_dim=1)

# Forward pass
x = Tensor(np.random.randn(1, 10))
hidden = activation(layer1(x))
output = layer2(hidden)
```

### Optimizers

```python
from machinetypes import SGD, Adam

# Initialize model parameters
params = [layer1.parameters(), layer2.parameters()]

# Create optimizers
sgd_optimizer = SGD(params, lr=0.01, momentum=0.9)
adam_optimizer = Adam(params, lr=0.001)
```

### Loss Functions

```python
from machinetypes import MSELoss, CrossEntropyLoss

# Initialize loss functions
mse_loss = MSELoss()
ce_loss = CrossEntropyLoss()

# Compute loss
loss = mse_loss(predictions, targets)
```

### Metrics

```python
from machinetypes import Accuracy

# Initialize metric
accuracy = Accuracy()

# Update metric
accuracy.update(predictions, targets)
print(f'Accuracy: {accuracy.compute():.2%}')
```

## Documentation

For detailed documentation, please visit [Documentation Link].

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
