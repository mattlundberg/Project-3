import tensorflow as tf
import os
from typing import Any, Optional, Union

class ModelHelper:
    """Helper class for saving and loading TensorFlow models"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ModelHelper
        
        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def save_model(self, model: tf.keras.Model, filename: str) -> bool:
        """
        Save a TensorFlow model to disk
        
        Args:
            model: The TensorFlow model to save
            filename: Name of the file to save the model to
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            filepath = os.path.join(self.model_dir, filename)
            model.save(filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, filename: str) -> Optional[tf.keras.Model]:
        """
        Load a TensorFlow model from disk
        
        Args:
            filename: Name of the file to load the model from
            
        Returns:
            The loaded TensorFlow model if successful, None otherwise
        """
        try:
            filepath = os.path.join(self.model_dir, filename)
            model = tf.keras.models.load_model(filepath)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    def list_models(self) -> list:
        """
        List all model files in the model directory
        
        Returns:
            List of model filenames
        """
        try:
            return [f for f in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, f))]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []
            
    def delete_model(self, filename: str) -> bool:
        """
        Delete a model directory
        
        Args:
            filename: Name of the model directory to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                tf.keras.backend.clear_session()
                import shutil
                shutil.rmtree(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting model: {str(e)}")
            return False


    def build_model(self, model_config: dict) -> Union[object, None]:
        """
        Build a new TensorFlow model based on specified configuration
        
        Args:
            model_config: Dictionary containing model architecture specifications
                Required keys:
                - layer_sizes: List of integers specifying nodes per dense layer
                - activation: Activation function to use (e.g. 'relu', 'sigmoid')
                Optional keys:
                - dropout_rate: Float between 0-1 for dropout layers
                - learning_rate: Float for optimizer learning rate
                - optimizer: String specifying optimizer ('adam', 'sgd', etc.)
            
        Returns:
            Built TensorFlow model if successful, None otherwise
        """
        try:
            
            # Extract config parameters
            layer_sizes = model_config['layer_sizes']
            activation = model_config.get('activation', 'relu')
            dropout_rate = model_config.get('dropout_rate', 0.2)
            learning_rate = model_config.get('learning_rate', 0.001)
            optimizer = model_config.get('optimizer', 'adam')
            
            # Build sequential model
            model = tf.keras.Sequential()
            
            # Add input layer
            model.add(tf.keras.layers.Dense(layer_sizes[0], activation=activation))
            
            # Add hidden layers with dropout
            for units in layer_sizes[1:-1]:
                model.add(tf.keras.layers.Dense(units, activation=activation))
                model.add(tf.keras.layers.Dropout(dropout_rate))
                
            # Add output layer
            model.add(tf.keras.layers.Dense(layer_sizes[-1]))
            
            # Configure optimizer
            if optimizer.lower() == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
                
            # Compile model
            model.compile(optimizer=opt, loss='mse')
            
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            return None