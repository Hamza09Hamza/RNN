import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from LogisticRegression import LogisticRegression
import time

class MulticlassLogisticRegression(BaseEstimator, ClassifierMixin):
  
    def __init__(
        self,
        alpha=0.01,
        iterations=1000,
        use_l2=False,
        lambda_=0.01,
        use_decay=False,
        decay=0.0,
        early_stopping=False,
        tol=1e-4,
        verbose=True
    ):
        self.alpha = alpha
        self.iterations = iterations
        self.use_l2 = use_l2
        self.lambda_ = lambda_
        self.use_decay = use_decay
        self.decay = decay
        self.early_stopping = early_stopping
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X, y):

        # Input validation
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ <= 1:
            raise ValueError(
                "This classifier requires at least two classes for training. "
                f"Found only {self.n_classes_} class."
            )
        
        # Initialize dictionary to store binary classifiers
        self.models_ = {}
        
        # Track training time
        start_time = time.time()
        
        if self.verbose:
            print(f"Training multi-class classifier with {self.n_classes_} classes")
            print("=" * 50)
        
        # One-vs-Rest approach: train one classifier per class
        for i, cls in enumerate(self.classes_):
            if self.verbose:
                print(f"Training classifier for class {cls} ({i+1}/{self.n_classes_})")
            
            # Create binary targets (1 for current class, 0 for others)
            y_binary = (y == cls).astype(int)
            
            # Train binary classifier
            model = LogisticRegression(
                alpha=self.alpha, 
                iterations=self.iterations,
                use_l2=self.use_l2, 
                lambda_=self.lambda_,
                use_decay=self.use_decay, 
                decay=self.decay,
                early_stopping=self.early_stopping, 
                tol=self.tol
            )
            model.fit(X, y_binary)
            
            # Store trained model
            self.models_[cls] = model
            
            if self.verbose:
                # Calculate accuracy for this binary classifier
                y_pred = model.predict(X)
                acc = accuracy_score(y_binary, y_pred)
                print(f"  - Training accuracy for class {cls}: {acc:.4f}")
        
        # Calculate total training time
        self.training_time_ = time.time() - start_time
        
        if self.verbose:
            print("=" * 50)
            print(f"Training complete in {self.training_time_:.2f} seconds")
        
        # Return the classifier
        return self
    
    def predict_proba(self, X):

        # Check if fit has been called
        check_is_fitted(self, ["classes_", "models_"])
            
        # Input validation
        X = check_array(X)
        
        # Initialize probability matrix
        probs = np.zeros((X.shape[0], self.n_classes_))
        
        # Get binary probabilities from each model
        for i, cls in enumerate(self.classes_):
            probs[:, i] = self.models_[cls].predict_proba(X)
        
        # Apply softmax to normalize probabilities
        # First, stabilize the numerical range to prevent overflow/underflow
        probs_exp = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs_normalized = probs_exp / np.sum(probs_exp, axis=1, keepdims=True)
        
        return probs_normalized
    
    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ["classes_", "models_"])
            
        # Input validation
        X = check_array(X)
        
        # Get probabilities for each class
        probs = self.predict_proba(X)
        
        # Return class with highest probability
        return self.classes_[np.argmax(probs, axis=1)]
    
    def decision_function(self, X):

        # For compatibility with scikit-learn metrics that expect a decision_function
        return self.predict_proba(X)
    
    def score(self, X, y):

        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):

        return {
            'alpha': self.alpha,
            'iterations': self.iterations,
            'use_l2': self.use_l2,
            'lambda_': self.lambda_,
            'use_decay': self.use_decay,
            'decay': self.decay,
            'early_stopping': self.early_stopping,
            'tol': self.tol,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)
        return self 