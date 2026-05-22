"""
Prediction utilities for the MNIST CNN project.

Run random test predictions:
    python predict.py --random --count 5

Run prediction on your own image:
    python predict.py --image images/my_digit.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "mnist_cnn_model.h5"
RANDOM_PREDICTIONS_PATH = BASE_DIR / "images" / "random_predictions.png"
CUSTOM_PREDICTION_PATH = BASE_DIR / "images" / "custom_prediction.png"


def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved CNN model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: python train.py"
        )
    return load_model(model_path)


def load_test_data():
    """Load and preprocess only the MNIST test data."""
    (_train_images, _train_labels), (test_images, test_labels) = mnist.load_data()
    test_images = test_images.astype("float32") / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1)
    return test_images, test_labels


def prepare_pil_image(image):
    """
    Prepare a Pillow image so it looks like MNIST data.

    MNIST images are white digits on a black background. This function converts
    common custom images into that style, resizes them, and centers them.
    """
    image = image.convert("L")
    image = ImageOps.autocontrast(image)

    image_array = np.array(image)

    # If the image has a light background with dark writing, invert it.
    if image_array.mean() > 127:
        image = ImageOps.invert(image)

    # Crop empty background around the digit, then resize and center it.
    bounding_box = image.getbbox()
    if bounding_box:
        image = image.crop(bounding_box)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    canvas.paste(image, (left, top))

    model_input = np.array(canvas).astype("float32") / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)

    return model_input, canvas


def prepare_custom_image(image_path):
    """Open a custom digit image from disk and prepare it for prediction."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)

    # Handle transparent PNGs by placing them on a white background first.
    if image.mode == "RGBA":
        white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        white_background.alpha_composite(image)
        image = white_background.convert("L")

    return prepare_pil_image(image)


def predict_random_test_images(count=5, model_path=DEFAULT_MODEL_PATH):
    """Predict random unseen images from the MNIST test set and display them."""
    model = load_trained_model(model_path)
    test_images, test_labels = load_test_data()

    count = max(1, min(count, len(test_images)))
    random_indices = np.random.choice(len(test_images), count, replace=False)
    selected_images = test_images[random_indices]
    selected_labels = test_labels[random_indices]

    predictions = model.predict(selected_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(count * 2.5, 3))
    for index in range(count):
        plt.subplot(1, count, index + 1)
        plt.imshow(selected_images[index].reshape(28, 28), cmap="gray")

        actual_label = selected_labels[index]
        predicted_label = predicted_labels[index]
        color = "green" if actual_label == predicted_label else "red"

        plt.title(
            f"Actual: {actual_label}\nPredicted: {predicted_label}",
            color=color,
        )
        plt.axis("off")

    plt.tight_layout()
    RANDOM_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(RANDOM_PREDICTIONS_PATH)
    print(f"Random prediction image saved to: {RANDOM_PREDICTIONS_PATH}")
    plt.show()


def predict_custom_image(image_path, model_path=DEFAULT_MODEL_PATH):
    """Predict a digit from a custom image on local storage."""
    model = load_trained_model(model_path)
    model_input, display_image = prepare_custom_image(image_path)

    prediction = model.predict(model_input, verbose=0)[0]
    predicted_label = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    print(f"Predicted Digit: {predicted_label}")
    print(f"Confidence: {confidence * 100:.2f}%")

    plt.figure(figsize=(4, 4))
    plt.imshow(display_image, cmap="gray")
    plt.title(f"Predicted: {predicted_label} ({confidence * 100:.2f}%)")
    plt.axis("off")
    CUSTOM_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CUSTOM_PREDICTION_PATH)
    print(f"Custom prediction image saved to: {CUSTOM_PREDICTION_PATH}")
    plt.show()

    return predicted_label, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digits with a CNN.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained .h5 model file.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Predict random unseen images from the MNIST test set.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of random test images to predict.",
    )
    parser.add_argument(
        "--image",
        help="Path to a custom handwritten digit image.",
    )

    args = parser.parse_args()

    if args.image:
        predict_custom_image(args.image, args.model)
    else:
        predict_random_test_images(args.count, args.model)


if __name__ == "__main__":
    main()
