"""
Train a Convolutional Neural Network on the MNIST handwritten digits dataset.

Run:
    python train.py

The trained model is saved to:
    models/mnist_cnn_model.h5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

DEFAULT_MODEL_PATH = MODELS_DIR / "mnist_cnn_model.h5"
TRAINING_PLOT_PATH = IMAGES_DIR / "training_accuracy_loss.png"
CONFUSION_MATRIX_PATH = IMAGES_DIR / "confusion_matrix.png"


def load_and_preprocess_data():
    """Load MNIST and prepare it for CNN training."""
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1.
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # CNNs expect images in this format: (samples, height, width, channels).
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # Convert labels such as 7 into one-hot format: [0,0,0,0,0,0,0,1,0,0].
    train_labels_one_hot = to_categorical(train_labels, 10)
    test_labels_one_hot = to_categorical(test_labels, 10)

    return (
        train_images,
        train_labels_one_hot,
        train_labels,
        test_images,
        test_labels_one_hot,
        test_labels,
    )


def build_cnn_model():
    """Create and compile a beginner-friendly CNN model."""
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history, output_path=TRAINING_PLOT_PATH):
    """Save accuracy and loss graphs after training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Training graph saved to: {output_path}")


def plot_confusion_matrix(actual_labels, predicted_labels, output_path=CONFUSION_MATRIX_PATH):
    """Create and save a confusion matrix image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(actual_labels, predicted_labels)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    digit_labels = np.arange(10)
    plt.xticks(digit_labels, digit_labels)
    plt.yticks(digit_labels, digit_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Write each number inside the matrix cells.
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            plt.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color=color,
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def train_model(epochs=5, batch_size=128, model_path=DEFAULT_MODEL_PATH):
    """Train, evaluate, and save the CNN model."""
    (
        train_images,
        train_labels_one_hot,
        _train_labels,
        test_images,
        test_labels_one_hot,
        test_labels,
    ) = load_and_preprocess_data()

    model = build_cnn_model()
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        train_images,
        train_labels_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
    )

    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    plot_training_history(history)

    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))

    plot_confusion_matrix(test_labels, predicted_labels)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST CNN model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained .h5 model.",
    )

    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, model_path=args.model)


if __name__ == "__main__":
    main()
