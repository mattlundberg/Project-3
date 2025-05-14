# Standard library imports
import os
import re
import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

# Third party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize
import tensorboard as notebook
from sentence_transformers import SentenceTransformer

class ModelHelper:
    """Helper class for managing TensorFlow models"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ModelHelper
        
        Args:
            model_dir: Directory to store model files
        """
        self.truthfulness_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
        self.model_dir = model_dir
        self.vectorizer = None  # Will store the trained TextVectorization layer
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def normalize_counts(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize truthfulness counts using sklearn's preprocessing.
        
        Args:
            df: DataFrame containing truthfulness columns
            
        Returns:
            Normalized counts as numpy array
        """
        counts = df[self.truthfulness_columns].values
        # L1 normalization (sum of values = 1) with built-in zero handling
        return normalize(counts, norm='l1', axis=1)
      
    def prepare_datasets(
        self,
        train_sequences: np.ndarray,
        train_labels: np.ndarray,
        val_sequences: np.ndarray,
        val_labels: np.ndarray,
        test_sequences: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare TensorFlow datasets for training, validation, and testing.
        
        Args:
            train_sequences: Training text sequences
            train_labels: Training labels
            val_sequences: Validation text sequences
            val_labels: Validation labels
            test_sequences: Test text sequences
            test_labels: Test labels
            batch_size: Size of training batches
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
        
        # Batch and prefetch the datasets
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
            
    def create_text_classification_model(self,
                                       num_classes: int = 5) -> tf.keras.Model:
        """
        Create a model for text classification with 5 truthfulness categories using sentence embeddings
        
        Args:
            num_classes: Number of output classes (5 for truthfulness categories)
            
        Returns:
            Compiled TensorFlow model
        """
        # Initialize a sequential model for text classification
        model = tf.keras.Sequential([
            # Input layer for sentence embeddings (384 dimensions from all-MiniLM-L6-v2)
            tf.keras.layers.Input(shape=(384,), dtype=tf.float32),
            
            # First dense layer with residual connection
            tf.keras.layers.Dense(768, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Second dense layer with residual connection
            tf.keras.layers.Dense(512, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Third dense layer
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Fourth dense layer
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Final output layer with softmax activation
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Use a lower learning rate with cosine decay
        initial_learning_rate = 0.0005
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0001
        )
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                    tf.keras.metrics.Precision(name='precision', class_id=None),
                    tf.keras.metrics.Recall(name='recall', class_id=None)]
        )
        
        return model
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing non-letter characters and converting to lowercase.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r'[^a-zA-Z\s ]', '', text.lower())

    def create_vectorizer(self, texts: List[str], max_tokens: int = 10000, 
                         max_sequence_length: int = 200) -> None:
        """
        Create and fit a TextVectorization layer on the training data.
        
        Args:
            texts: List of text strings to fit the vectorizer on
            max_tokens: Maximum number of tokens in the vocabulary
            max_sequence_length: Maximum length of sequences
        """
        # Clean texts first
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Create and fit the vectorizer
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            output_sequence_length=max_sequence_length
        )
        self.vectorizer.adapt(cleaned_texts)

    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the trained TextVectorization layer to a file.
        
        Args:
            filepath: Path where the vectorizer will be saved
            
        Raises:
            ValueError: If vectorizer hasn't been trained yet
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not trained. Call create_vectorizer first.")
            
        # Create a model containing just the vectorizer layer
        vectorizer_model = tf.keras.Sequential([self.vectorizer])
        
        # Save the model containing the vectorizer
        vectorizer_model.save(filepath, save_format='tf')

    def load_vectorizer(self, filepath: str) -> None:
        """
        Load a previously saved TextVectorization layer from a file.
        
        Args:
            filepath: Path to the saved vectorizer file
            
        Raises:
            FileNotFoundError: If the vectorizer file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No vectorizer found at {filepath}")
            
        # Load the model containing the vectorizer
        loaded_model = tf.keras.models.load_model(filepath)
        
        # Extract the vectorizer layer
        self.vectorizer = loaded_model.layers[0]

    def preprocess_text(self,
                       texts: Union[str, List[str]],
                       return_tokens: bool = False) -> Union[np.ndarray, List[str]]:
        """
        Preprocess text data using the sentence transformer.
        
        Args:
            texts: Single text string or list of text strings
            return_tokens: If True, returns cleaned tokens instead of encoded sequences
            
        Returns:
            If return_tokens is False: Encoded text data as numpy array
            If return_tokens is True: List of cleaned tokens
        """
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        if return_tokens:
            # Return cleaned tokens
            return [token for text in cleaned_texts for token in text.split() if token]
        
        # Use sentence transformer to encode the texts
        # This will return a numpy array of shape (n_texts, embedding_dim)
        encoded_texts = self.sentence_transformer.encode(cleaned_texts, 
                                                       convert_to_tensor=False,
                                                       show_progress_bar=False)
        return encoded_texts
        
    def create_model(self, 
                    input_shape: tuple,
                    layer_sizes: List[int],
                    activation: str = 'relu',
                    output_activation: str = 'sigmoid',
                    dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Create a new TensorFlow model with specified architecture
        
        Args:
            input_shape: Shape of input data (excluding batch size)
            layer_sizes: List of neurons per layer
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled TensorFlow model
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Hidden layers
        for units in layer_sizes[:-1]:
            model.add(tf.keras.layers.Dense(units, activation=activation))
            model.add(tf.keras.layers.Dropout(dropout_rate))
            
        # Output layer
        model.add(tf.keras.layers.Dense(layer_sizes[-1], activation=output_activation))
        
        return model
        
    def compile_model(self, 
                     model: tf.keras.Model,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'binary_crossentropy',
                     metrics: List[str] = ['accuracy']) -> tf.keras.Model:
        """
        Compile a TensorFlow model with specified parameters
        
        Args:
            model: Model to compile
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
            
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model
        
    def save_model(self, model: tf.keras.Model, filename: str) -> bool:
        """
        Save a TensorFlow model and its associated vectorizer to disk
        
        Args:
            model: The TensorFlow model to save
            filename: Name of the file to save the model to (without extension)
            
        Returns:
            bool: True if save successful, False otherwise
            
        Raises:
            ValueError: If vectorizer hasn't been trained yet
        """
        try:
            # Ensure filename has .keras extension
            if not filename.endswith('.keras'):
                filename = f"{filename}.keras"
            model_path = os.path.join(self.model_dir, filename)
            
            # Save the model
            model.save(model_path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, filename: str) -> Optional[tf.keras.Model]:
        """
        Load a TensorFlow model and its associated vectorizer from disk
        
        Args:
            filename: Name of the file to load the model from (without extension)
            
        Returns:
            The loaded TensorFlow model if successful, None otherwise
            
        Raises:
            FileNotFoundError: If either the model or vectorizer file doesn't exist
        """
        try:
            # Load the model
            model_path = os.path.join(self.model_dir, f"{filename}.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model found at {model_path}")
            model = tf.keras.models.load_model(model_path)
            
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
            
    def train_model(self,
                   model: tf.keras.Model,
                   train_data: tf.data.Dataset,
                   validation_data: Optional[tf.data.Dataset] = None,
                   epochs: int = 10,
                   batch_size: int = 32,
                   callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """
        Train a TensorFlow model
        
        Args:
            model: Model to train
            train_data: Training dataset
            validation_data: Optional validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of callbacks to use during training
            
        Returns:
            Training history
        """

        # Create a TensorBoard callback
        log_dir = os.path.join(self.model_dir, 'logs', 'fit', 
                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True) 
            
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.2 if validation_data is None else None
        )
        
        return history
        
    def evaluate_model(self,
                      model: tf.keras.Model,
                      test_data: tf.data.Dataset,
                      batch_size: int = 32,
                      verbose: int = 1,
                      return_samples: bool = False,
                      sample_size: int = 100) -> Dict[str, Any]:
        """
        Evaluate a trained model with detailed metrics and optional sample predictions
        
        Args:
            model: Model to evaluate
            test_data: Test dataset
            batch_size: Batch size for evaluation
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
            return_samples: Whether to return sample predictions for manual verification
            sample_size: Number of samples to return if return_samples is True
            
        Returns:
            Dictionary containing:
            - Standard evaluation metrics (loss, accuracy etc.)
            - Confusion matrix if classification task
            - Sample predictions if return_samples=True
            - Additional metrics like F1 score, precision, recall for classification
        """
        # Get base evaluation metrics
        metrics = model.evaluate(test_data, batch_size=batch_size, verbose=verbose, return_dict=True)
        
        # Get predictions for additional analysis
        predictions = model.predict(test_data, batch_size=batch_size, verbose=0)
        
        # Add confusion matrix for classification tasks
        if model.output_shape[-1] > 1:  # Multi-class
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.concatenate([y for x, y in test_data], axis=0)
            metrics['confusion_matrix'] = tf.math.confusion_matrix(y_true, y_pred).numpy().tolist()
            
            # Add classification metrics
            metrics['precision'] = tf.keras.metrics.Precision()(y_true, y_pred).numpy()
            metrics['recall'] = tf.keras.metrics.Recall()(y_true, y_pred).numpy()
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        # Add sample predictions if requested
        if return_samples:
            sample_indices = np.random.choice(len(predictions), min(sample_size, len(predictions)), replace=False)
            metrics['sample_predictions'] = predictions[sample_indices].tolist()
            metrics['sample_true_values'] = np.concatenate([y for x, y in test_data], axis=0)[sample_indices].tolist()
            
        return metrics
        
    def predict(self,
               model: tf.keras.Model,
               data: Union[np.ndarray, tf.data.Dataset]) -> np.ndarray:
        """
        Make predictions using the model
        
        Args:
            model: Model to use for predictions
            data: Input data for predictions
            
        Returns:
            Model predictions
        """
        return model.predict(data)
        
    def list_models(self) -> List[str]:
        """
        List all saved models in the model directory
        
        Returns:
            List of model names (without .keras extension)
        """
        try:
            # Get all .keras files in the model directory
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.endswith('.keras') and f != 'vectorizer.keras']
            # Remove the .keras extension from the filenames
            return [os.path.splitext(f)[0] for f in model_files]
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []
            
    def delete_model(self, filename: str) -> bool:
        """
        Delete a saved model
        
        Args:
            filename: Name of the model to delete
            
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