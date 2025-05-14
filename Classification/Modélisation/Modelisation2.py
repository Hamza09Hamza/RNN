import platform
import subprocess
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

Path_Data='Classification/Data/Processed/'
X_train = pd.read_csv(Path_Data+'X_train.csv')
X_val = pd.read_csv(Path_Data+'X_val.csv')
X_test = pd.read_csv(Path_Data+'X_test.csv')

y_train = pd.read_csv(Path_Data+'y_train.csv')
y_val = pd.read_csv(Path_Data+'y_val.csv')
y_test = pd.read_csv(Path_Data+'y_test.csv')


import xgboost 
import catboost
import lightgbm as lgb

models = {
    # "XGBoost": xgboost.XGBClassifier(),
    "CatBoost": catboost.CatBoostClassifier(verbose=False),
    "LightGBM": lgb.LGBMClassifier()
}

param_grids = {
    # "XGBoost": {
    #     'max_depth': [7],
    #     'learning_rate': [0.2],
    #     'n_estimators': [200]
    # },
    "CatBoost": {
        'depth': [7],
        'iterations': [1000],
        'learning_rate': [0.2]
    },
    "LightGBM": {
        'learning_rate': [0.1],
        'n_estimators': [100],
        'num_leaves': [100]
    }
}


def detect_environment():
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Apple Silicons (M1/M2)
    if system == 'darwin' and 'arm' in machine:
        return "M1"

    # CUDA-compatible NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return "CUDA"
    except FileNotFoundError:
        pass

    return "CPU"

env = detect_environment()
print(f" Environnement détecté : {env}")

# Create results directory if it doesn't exist
results_dir = "./model_results"
os.makedirs(results_dir, exist_ok=True)

# Create directory for learning curve plots
plots_dir = os.path.join(results_dir, "learning_curves")
os.makedirs(plots_dir, exist_ok=True)

# Initialize results file
results_file = os.path.join(results_dir, "model_results.json")
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump({"models": []}, f)

# Load existing results
with open(results_file, 'r') as f:
    all_results = json.load(f)

# Track best overall model
best_overall_model = {
    "model_name": None,
    "best_params": None,
    "val_accuracy": 0.0
}

def save_model_result(model_name, best_params, val_accuracy):
    """Save the result of a single model to the JSON file"""
    model_result = {
        "model_name": model_name,
        "best_params": best_params,
        "val_accuracy": val_accuracy
    }
    
    # Update all_results
    all_results["models"].append(model_result)
    
    # Update best overall model
    global best_overall_model
    if val_accuracy > best_overall_model["val_accuracy"]:
        best_overall_model = model_result.copy()
    
    # Save to file
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✅ Results for {model_name} saved to {results_file}")

def wait_for_input(model_name):
    """Wait for user input before proceeding to next model"""
    print(f"\n{'='*50}")
    print(f"Completed training for {model_name}")
    print("Press Enter to continue to next model or 'q' to quit...")
    user_input = input()
    if user_input.lower() == 'q':
        print("\nExiting early...")
        print_summary()
        exit()

def print_summary():
    """Print summary of all results"""
    print("\n\n📊 Final Summary of Results:")
    for model in all_results["models"]:
        print(f"\n{model['model_name']}:")
        print(f"  Validation Accuracy: {model['val_accuracy']:.4f}")
        print(f"  Best Parameters: {model['best_params']}")
    
    print("\n🏆 Best Overall Model:")
    if best_overall_model["model_name"]:
        print(f"{best_overall_model['model_name']} with accuracy {best_overall_model['val_accuracy']:.4f}")
        print(f"Parameters: {best_overall_model['best_params']}")
    else:
        print("No models completed yet.")

def setup_gpu_for_model(model, model_name):
    """Properly configure GPU settings for each model"""
    if env == "CUDA":
        if model_name == "XGBoost":
            model.set_params(
                tree_method='hist',
                device='cuda:0'
            )
        elif model_name == "LightGBM":
            model.set_params(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0
            )
        elif model_name == "CatBoost":
            model.set_params(
                task_type='GPU',
                devices='0:0',
                verbose=0
            )
    elif env == "M1":
        if model_name == "LightGBM":
            model.set_params(device_type='metal')
        elif model_name == "CatBoost":
            model.set_params(task_type='CPU', verbose=0)
    else:  # CPU fallback
        if model_name == "XGBoost":
            model.set_params(tree_method='hist', device='cpu')
        elif model_name == "LightGBM":
            model.set_params(device_type='cpu')
    
    return model

# Function to generate learning curves
def generate_learning_curve(estimator, X, y, model_name, cv=5):
    """
    Generate learning curves to evaluate model performance with varying training set sizes
    
    Parameters:
    -----------
    estimator : estimator object
        The model with the best parameters from grid search
    X : array-like
        Feature data
    y : array-like
        Target data
    model_name : str
        Name of the model for saving the plot
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    None, but saves the learning curve plot
    """
    print(f"\n📉 Generating learning curve for {model_name}...")
    
    # Configure train sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Generate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, 
        train_sizes=train_sizes,
        cv=cv, 
        scoring='accuracy',
        n_jobs=1,  # Use 1 for GPU models
        verbose=1
    )
    
    # Calculate mean and std for training and validation scores
    train_mean = 1 - np.mean(train_scores, axis=1)  # Convert to error rate
    train_std = np.std(train_scores, axis=1)
    
    val_mean = 1 - np.mean(val_scores, axis=1)  # Convert to error rate
    val_std = np.std(val_scores, axis=1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Error Rate (1 - Accuracy)')
    plt.grid()
    
    # Plot error rate
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Error')
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Validation Error')
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color='g'
    )
    
    plt.legend(loc='best')
    
    # Save plot
    plot_file = os.path.join(plots_dir, f"{model_name}_learning_curve.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"✅ Learning curve saved to {plot_file}")
    
    # Analyze the learning curve
    analyze_learning_curve(train_mean, val_mean, model_name)

def analyze_learning_curve(train_errors, val_errors, model_name):
    """Analyze the learning curve to diagnose if the model is underfitting or overfitting"""
    final_train_error = train_errors[-1]
    final_val_error = val_errors[-1]
    error_gap = final_val_error - final_train_error
    
    print("\n🔍 Learning Curve Analysis:")
    print(f"  - Training Error Rate: {final_train_error:.4f}")
    print(f"  - Validation Error Rate: {final_val_error:.4f}")
    print(f"  - Error Gap: {error_gap:.4f}")
    
    # Analyze variance/bias trade-off
    if final_train_error > 0.1:  # Arbitrary threshold of 10% error
        print("  - High training error suggests UNDERFITTING (high bias)")
    
    if error_gap > 0.1:  # Arbitrary threshold of 10% gap
        print("  - Large gap between training and validation errors suggests OVERFITTING (high variance)")
    
    if final_train_error <= 0.1 and error_gap <= 0.1:
        print("  - Model appears well balanced (good bias-variance trade-off)")
    
    # Check if learning curve is still improving
    if val_errors[0] - val_errors[-1] < 0.05:  # Less than 5% improvement
        print("  - Learning curve shows minimal improvement with more data")
        print("  - The model might benefit from different features or approaches")
    else:
        print("  - Learning curve shows good improvement with more data")
        if val_errors[-2] - val_errors[-1] > 0.01:  # Still improving
            print("  - Model might benefit from even more training data")

# Add this function to monitor GPU usage
def monitor_gpu():
    try:
        if env == "CUDA":
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
            print("GPU Usage:\n", result.stdout.decode())
    except:
        print("Could not monitor GPU usage")

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🚀 Starting GridSearch for: {model_name}")
    
    if model_name == "Logistic Regression":
        print("  GridSearch skipped for custom Logistic Regression.")
        continue

    # Proper GPU setup
    model = setup_gpu_for_model(model, model_name)
    
    # Monitor before training
    monitor_gpu()
    
    try:
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            cv=3,
            n_jobs=1,  # Keep this as 1 for GPU models
            verbose=2
        )
        
        # Prepare data - convert to numpy arrays properly
        if env == "CUDA" and model_name in ["XGBoost"]:
            X_train_gpu = X_train.values.astype(np.float32)  # Proper conversion
            y_train_gpu = y_train.values.astype(np.float32).ravel()
            
            # For validation data too
            X_val_gpu = X_val.values.astype(np.float32)
            y_val_gpu = y_val.values.astype(np.float32).ravel()
            
            grid_search.fit(X_train_gpu, y_train_gpu)
            best_model = grid_search.best_estimator_
            y_pred_val = best_model.predict(X_val_gpu)  # Predict on GPU data
            val_acc = accuracy_score(y_val_gpu, y_pred_val)
            
            # Generate learning curve using the best model
            # Combine train and validation data for learning curve
            X_combined = np.vstack((X_train_gpu, X_val_gpu))
            y_combined = np.concatenate((y_train_gpu, y_val_gpu))
            generate_learning_curve(best_model, X_combined, y_combined, model_name)
            
        else:
            # For CPU models
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred_val = best_model.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred_val)
            
            # Generate learning curve using the best model
            # Combine train and validation data for learning curve
            X_combined = pd.concat([X_train, X_val])
            y_combined = pd.concat([y_train, y_val])
            generate_learning_curve(best_model, X_combined, y_combined, model_name)

        # Monitor after training
        monitor_gpu()
        
        print(f"\n🎯 Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"📊 {model_name} - Validation Accuracy: {val_acc:.4f}")

        save_model_result(model_name, grid_search.best_params_, val_acc)
        
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        save_model_result(model_name, {"error": str(e)}, 0.0)
    
    # Clean up GPU memory
    if env == "CUDA":
        import gc
        del grid_search
        gc.collect()
    
    wait_for_input(model_name)

# Print final summary
print_summary()