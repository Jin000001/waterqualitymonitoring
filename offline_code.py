"""Model definitions and training utilities for CAE and classification baselines."""

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Model


TIME_STEPS_CNN = 48
FEATURES_CNN = 1
N_CLASSES = 4
TIME_STEPS = 336


def create_cae(input_shape):
    """Create and compile a 1D convolutional autoencoder."""
    _, n_features = input_shape
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(16, 7, activation="relu", padding="same", strides=2)(inputs)
    x = layers.Conv1D(32, 7, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1D(64, 7, activation="relu", padding="same", strides=2)(x)
    encoded = layers.Conv1D(128, 7, activation="relu", padding="same", strides=2)(x)

    x = layers.Conv1DTranspose(64, 7, activation="relu", padding="same", strides=2)(encoded)
    x = layers.Conv1DTranspose(32, 7, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1DTranspose(16, 7, activation="relu", padding="same", strides=2)(x)
    decoded = layers.Conv1DTranspose(
        n_features, 7, activation="linear", padding="same", strides=2
    )(x)

    cae = models.Model(inputs, decoded)
    cae.compile(optimizer="adam", loss="mse")
    return cae


def create_cnnlstm(input_shape):
    """Create and compile a CNN-LSTM classifier."""
    input_layer = Input(shape=input_shape)

    x = Conv1D(filters=64, kernel_size=11, activation="relu", padding="same")(input_layer)
    x = MaxPooling1D(pool_size=4, padding="same")(x)
    x = LSTM(units=64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(units=32, activation="relu")(x)
    output_layer = Dense(units=N_CLASSES, activation="softmax")(x)

    cnnlstm = Model(inputs=input_layer, outputs=output_layer)
    cnnlstm.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return cnnlstm


def create_cnn(input_shape):
    """Create and compile a CNN classifier."""
    input_layer = Input(shape=input_shape)

    x = Conv1D(filters=10, kernel_size=11, activation="relu")(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=20, kernel_size=5, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(units=100, activation="relu")(x)
    output_layer = Dense(units=5, activation="softmax")(x)

    cnn = Model(inputs=input_layer, outputs=output_layer)
    cnn.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return cnn


def create_lstm(input_shape):
    """Create and compile an LSTM classifier."""
    input_layer = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(input_layer)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    output_layer = Dense(N_CLASSES, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_mlp(input_shape):
    """Create and compile an MLP classifier."""
    input_layer = Input(shape=input_shape)

    x = Flatten()(input_layer)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output_layer = Dense(N_CLASSES, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_cae(x_train, input_shape):
    """Train a CAE and compute per-feature reconstruction thresholds."""
    n_samples = x_train.shape[0]
    cae_model = create_cae(input_shape)
    cae_model.summary()
    cae_model.fit(x_train, x_train, epochs=200, batch_size=64)

    decoded = cae_model.predict(x_train)
    n_features = input_shape[1]
    feature_losses = [[] for _ in range(n_features)]

    print(f"x_train shape: {x_train.shape}")
    print(f"decoded shape: {decoded.shape}")

    for i in range(n_samples):
        for j in range(n_features):
            reconstruction_loss = np.mean(np.square(x_train[i, :, j] - decoded[i, :, j]))
            feature_losses[j].append(reconstruction_loss)

    threshold_losses = [1.25 * max(losses) for losses in feature_losses]
    print("Initial threshold losses", threshold_losses)

    return cae_model, threshold_losses, decoded


def train_cnnlstm(x_train, y_train, input_shape):
    """Train a CNN-LSTM classifier."""
    model = create_cnnlstm(input_shape)
    model.fit(x_train, y_train, epochs=50, batch_size=64)

    loss, accuracy = model.evaluate(x_train, y_train)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return model


def train_cnn(x_train, y_train, input_shape):
    """Train a CNN classifier."""
    model = create_cnn(input_shape)
    model.fit(x_train, y_train, epochs=50, batch_size=64)

    loss, accuracy = model.evaluate(x_train, y_train)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return model


def train_model(model_type, x_train, y_train, input_shape):
    """Train a classifier selected by model type."""
    if model_type == "lstm":
        model = create_lstm(input_shape)
    elif model_type == "mlp":
        model = create_mlp(input_shape)
    elif model_type == "cnn":
        model = create_cnn(input_shape)
    elif model_type == "cnnlstm":
        model = create_cnnlstm(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Training {model_type}...")
    model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1)

    return model
