import tensorflow as tf
import os
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import datetime
import tensorboard as notebook
import pandas as pd

class ModelHelper:
    """Helper class for managing TensorFlow models"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize ModelHelper
        
        Args:
            model_dir: Directory to store model files
        """
        self.truthfulness_columns = ['barely_true_counts', 'pants_on_fire_counts', 'mostly_true_counts', 'half_true_counts', 'false_counts']
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def normalize_counts(self, df):
        counts = df[self.truthfulness_columns].values
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        return counts / row_sums      
      
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
                                       vocab_size: int,
                                       embedding_dim: int = 400,  # Increased for better word representation
                                       max_sequence_length: int = 200,
                                       num_classes: int = 5) -> tf.keras.Model:
        """
        Create a model for text classification with 5 truthfulness categories
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding layer
            max_sequence_length: Maximum length of input sequences
            num_classes: Number of output classes (5 for truthfulness categories)
            
        Returns:
            Compiled TensorFlow model
        """
        # Initialize a sequential model for text classification
        model = tf.keras.Sequential([
            # Input layer that accepts sequences of integers (word indices) of fixed length
            # Shape: (batch_size, max_sequence_length)
            tf.keras.layers.Input(shape=(max_sequence_length,), dtype=tf.float32),
            
            # Embedding layer converts integer indices to dense vectors of size embedding_dim
            # L2 regularization helps prevent overfitting by penalizing large weights
            # Shape: (batch_size, max_sequence_length, embedding_dim) 
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    embeddings_regularizer=tf.keras.regularizers.l2(0.001)),
            
            # Parallel CNN layers to capture different n-gram patterns:
            # 3-gram patterns (local features spanning 3 words)
            # Shape: (batch_size, max_sequence_length, 256)
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),  # Normalize activations for stable training
            
            # 5-gram patterns (medium-range features spanning 5 words) 
            tf.keras.layers.Conv1D(256, 5, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            
            # 7-gram patterns (longer-range features spanning 7 words)
            tf.keras.layers.Conv1D(256, 7, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            
            # First Bidirectional LSTM processes sequences in both directions
            # return_sequences=True keeps temporal dimension for next layer
            # Shape: (batch_size, max_sequence_length, 2*128)
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),  # Randomly drop 30% of units to prevent overfitting
            
            # Second Bidirectional LSTM layer for higher-level sequence features
            # Shape: (batch_size, 2*64)
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Dense layers for high-level feature extraction
            # L2 regularization on weights helps prevent overfitting
            # Shape: (batch_size, 256)
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Second dense layer further reduces dimensionality
            # Shape: (batch_size, 128) 
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Final output layer with softmax activation for multi-class classification
            # Shape: (batch_size, num_classes)
            # Each output represents probability of text belonging to that truthfulness category
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model with adjusted learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                    tf.keras.metrics.Precision(name='precision', class_id=None),
                    tf.keras.metrics.Recall(name='recall', class_id=None)]
        )
        
        return model
        
    def preprocess_text(self,
                       texts: Union[str, List[str]],
                       max_sequence_length: int = 200,
                       return_tokens: bool = False) -> Union[np.ndarray, List[str]]:
        """
        Preprocess text data for model input with cleaning and tokenization options.
        
        Args:
            texts: Single text string or list of text strings
            max_sequence_length: Maximum length of sequences
            return_tokens: If True, returns cleaned tokens instead of vectorized sequences
            
        Returns:
            If return_tokens is False: Vectorized text data as numpy array
            If return_tokens is True: List of cleaned tokens
            
        Example:
            >>> preprocess_text("Hey @user123! Check out #AI", return_tokens=True)
            ['hey', 'user', 'check', 'out', 'ai']
            >>> preprocess_text(["Hey @user123!", "Check out #AI"])
            array([[1, 2, 3], [4, 5, 6]])  # Vectorized sequences
        """
        import re
        
        # Convert single string to list for consistent processing
        if isinstance(texts, str):
            texts = [texts]
            
        # Clean text using lambda function to keep only letters and spaces
        clean_text = lambda x: re.sub(r'[^a-zA-Z\s ]', '', str(x))
        
        # Clean and tokenize texts
        cleaned_texts = []
        for text in texts:
            # Ensure text is a string and clean it
            if not isinstance(text, str):
                text = str(text)
            cleaned_text = clean_text(text.lower())
            cleaned_texts.append(cleaned_text)
            
        if return_tokens:
            # Return cleaned tokens
            return [token for text in cleaned_texts for token in text.split() if token]
            
        # Create text vectorization layer
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=10000,
            output_mode='int',
            output_sequence_length=max_sequence_length
        )
        
        # Adapt the layer to the cleaned texts
        vectorize_layer.adapt(cleaned_texts)
        
        # Vectorize the texts and ensure correct shape
        sequences = vectorize_layer(cleaned_texts)
        return tf.cast(sequences, tf.float32)  # Convert to float32 for model input
        
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
        Save a TensorFlow model to disk
        
        Args:
            model: The TensorFlow model to save
            filename: Name of the file to save the model to (without extension)
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Ensure filename has .keras extension
            if not filename.endswith('.keras'):
                filename = f"{filename}.keras"
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

        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6,
                    mode='min'
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                )
            ]
            
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
        List all saved models
        
        Returns:
            List of model names
        """
        try:
            return [f for f in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, f))]
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