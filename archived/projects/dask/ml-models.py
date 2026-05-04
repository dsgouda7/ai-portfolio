# ml_models.py (enhanced visualization)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def manual_gradient_descent(X_train, y_train, X_test, y_test,
                            alpha=0.01, n_epochs=1000, print_every=50, plot_loss=True):
    """
    Multiple linear regression using gradient descent.
    Tracks loss over epochs and plots if desired.
    """
    X_train_np = X_train
    y_train_np = y_train.values.reshape(-1,1)
    X_test_np = X_test
    y_test_np = y_test.values.reshape(-1,1)
    
    # Add bias term
    X_b = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
    theta = np.zeros((X_b.shape[1], 1))
    
    losses = []
    epochs_recorded = []
    
    for epoch in range(n_epochs):
        predictions = X_b @ theta
        errors = predictions - y_train_np
        gradient = (1 / X_b.shape[0]) * (X_b.T @ errors)
        theta -= alpha * gradient
        
        if epoch % print_every == 0:
            loss = np.mean(errors**2) / 2
            losses.append(loss)
            epochs_recorded.append(epoch)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    # Plot loss curve
    if plot_loss:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_recorded, losses, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Gradient Descent Convergence")
        plt.grid(True)
        plt.show()
    
    # Predict on test set
    X_test_b = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])
    y_pred_test = X_test_b @ theta
    
    return theta, y_pred_test, epochs_recorded, losses
