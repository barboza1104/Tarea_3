# Neural Network Implementation for Binary Classification

Implementation of a 3-layer neural network from scratch for binary classification tasks, with manual backpropagation and gradient descent optimization.

## Key Features
- **Synthetic Dataset Generation**: Creates 2D Gaussian distributions for classification
- **Network Architecture**: 3 hidden layers (2 → 6 → 10 → 1 neurons)
- **Activation Functions**: 
  - ReLU for hidden layers
  - Sigmoid for output layer
- **Custom Training Loop**: Manual implementation of forward/backward propagation
- **Error Tracking**: MSE loss monitoring during training

## Installation & Dependencies
    pip install numpy matplotlib scikit-learn


- Python 3.6+
- NumPy 1.19+
- Matplotlib 3.3+
- scikit-learn 0.24+

## Usage
1. Generate synthetic data:
    X, Y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2)



2. Initialize network parameters:
    layers_dims =
    params = initialize_parameters_deep(layers_dims)



3. Train the model:
    for epoch in range(50000):
    output = train(X, 0.001, params)
    if epoch % 50 == 0:
    current_loss = mse(Y, output)
    error.append(current_loss)



## Network Components
### Activation Functions
- `sigmoid(x, derivate=False)`: Logistic function for output layer
- `relu(x, derivate=False)`: Rectified Linear Unit for hidden layers

### Loss Function
- `mse(y, y_hat, derivate=False)`: Mean Squared Error implementation

### Parameter Initialization
- He initialization variant: weights ∈ [-1, 1]
- Bias initialization: ∈ [-1, 1]

## Training Configuration
- **Learning Rate**: 0.001
- **Epochs**: 50,000
- **Batch Size**: Full batch (all samples)
- **Update Frequency**: Loss logging every 50 epochs

## Results Visualization
## Plot training error
    plt.plot(error)
    plt.title('Training Loss Evolution')
    plt.xlabel('Epochs (x50)')
    plt.ylabel('MSE Loss')

## Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    Z = train(np.c_[xx.ravel(), yy.ravel()], 0.001, params, training=False)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.title('Decision Boundary')


## Key Implementation Details
1. **Forward Propagation**:
   - Matrix multiplication with `np.matmul`
   - Sequential layer activation
   - Intermediate values stored in params dictionary

2. **Backward Propagation**:
   - Manual gradient computation
   - Chain rule implementation
   - Parameter updates using learning rate

3. **Memory Management**:
   - All intermediate values stored in dictionary
   - In-place operations for memory efficiency

## Performance Considerations
- **Training Time**: ~2-5 minutes on CPU (varies by hardware)
- **Convergence**: Typically reaches <0.05 MSE after 50k epochs
- **Vectorization**: Full batch processing for stability

## Extension Opportunities
1. Add mini-batch training capability
2. Implement different optimization algorithms (Adam, RMSprop)
3. Add regularization techniques (L2, dropout)
4. Create prediction interface for new data
5. Add cross-validation support
