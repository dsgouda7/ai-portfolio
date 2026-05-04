# ml_evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_regression(y_test, y_pred, plot=True):
    """
    Evaluate regression predictions.
    Prints MSE and R^2, optionally plots Actual vs Predicted with separate colors.
    
    Parameters:
        y_test : Dask Series or NumPy array
        y_pred : NumPy array (predictions)
        plot : bool, whether to plot graphs
    """
    # Convert Dask Series to NumPy array if necessary
    if hasattr(y_test, 'compute'):
        y_test_np = y_test.compute()
    else:
        y_test_np = y_test

    # Metrics
    mse = mean_squared_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    if plot:
        # Actual vs Predicted (line plot for clarity)
        plt.figure(figsize=(10,6))
        idx = np.arange(len(y_test_np))
        plt.plot(idx, y_test_np, 'o-', label='Actual', color='blue', alpha=0.7)
        plt.plot(idx, y_pred, 'x-', label='Predicted', color='orange', alpha=0.7)
        plt.xlabel("Sample Index")
        plt.ylabel("Target Value")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Residuals
        residuals = y_pred - y_test_np
        plt.figure(figsize=(10,6))
        plt.scatter(idx, residuals, alpha=0.7, color='purple')
        plt.hlines(0, 0, len(residuals), colors='r', linestyles='dashed')
        plt.xlabel("Sample Index")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.grid(True)
        plt.show()
