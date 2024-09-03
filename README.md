# SMS Spam Classification using Neural Networks

This project implements a neural network-based SMS spam classification system using TensorFlow and Keras. The goal is to accurately classify SMS messages as either "ham" (normal) or "spam" (unwanted advertisements or messages from companies).

## Project Overview

The project follows these main steps:

1. Data Preprocessing
2. Model Creation
3. Model Training
4. Message Prediction

## Data Preprocessing

- The dataset is loaded from TSV files containing labeled SMS messages.
- Text data is cleaned and tokenized using NLTK.
- Messages are converted to sequences of word indices and padded to ensure uniform length.
- Labels are converted to numerical values (0 for ham, 1 for spam).

Key concepts:

- Tokenization
- Sequence padding
- Text normalization

## Model Creation

A neural network is built using TensorFlow/Keras with the following architecture:

1. Embedding layer for learning word representations
2. LSTM layers for sequence processing
3. Dense layers with ReLU activation
4. Dropout for regularization
5. Final Dense layer with sigmoid activation for binary classification

Key concepts:

- Word embeddings
- Recurrent Neural Networks (LSTM)
- Dropout regularization

## Model Training

- Data is split into training and validation sets.
- The model is trained using binary cross-entropy loss and Adam optimizer.
- Class weights are applied to handle potential class imbalance.
- Training progress is monitored using accuracy and loss metrics.

Key concepts:

- Train-validation split
- Loss functions
- Optimization algorithms
- Class weighting

## Message Prediction

A `predict_message` function is implemented to:

1. Preprocess input messages
2. Use the trained model for prediction
3. Return the spam probability and corresponding label

## Evaluation

The model is evaluated on a test set, and performance metrics such as accuracy, precision, recall, and F1-score can be calculated.

## Potential Improvements

- Experiment with different model architectures (e.g., CNN, Transformer)
- Use advanced text preprocessing techniques (lemmatization, stemming)
- Implement data augmentation for text data
- Explore ensemble methods

This project demonstrates the application of natural language processing and deep learning techniques to solve a real-world problem of SMS spam classification.
