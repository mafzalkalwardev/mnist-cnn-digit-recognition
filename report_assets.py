"""
Generate image assets for the project report.

Run:
    python report_assets.py

Generated files:
    images/mnist_samples_0_to_9.png
    images/cnn_architecture_diagram.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

MNIST_SAMPLE_PATH = IMAGES_DIR / "mnist_samples_0_to_9.png"
CNN_DIAGRAM_PATH = IMAGES_DIR / "cnn_architecture_diagram.png"


def generate_mnist_sample_grid(output_path=MNIST_SAMPLE_PATH):
    """Save one MNIST sample image for each digit from 0 to 9."""
    (_train_images, _train_labels), (test_images, test_labels) = mnist.load_data()

    digit_images = []
    for digit in range(10):
        digit_index = np.where(test_labels == digit)[0][0]
        digit_images.append(test_images[digit_index])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 2.6))
    for digit, image in enumerate(digit_images):
        plt.subplot(1, 10, digit + 1)
        plt.imshow(image, cmap="gray")
        plt.title(str(digit))
        plt.axis("off")

    plt.suptitle("MNIST Dataset Samples: Digits 0 to 9", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"MNIST sample image saved to: {output_path}")


def generate_cnn_architecture_diagram(output_path=CNN_DIAGRAM_PATH):
    """Save a simple input-to-output CNN architecture diagram."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [
        ("Input", "28 x 28 x 1"),
        ("Conv2D", "32 filters\nReLU"),
        ("MaxPooling", "2 x 2"),
        ("Conv2D", "64 filters\nReLU"),
        ("MaxPooling", "2 x 2"),
        ("Flatten", "1D vector"),
        ("Dense", "128 neurons\nReLU"),
        ("Dropout", "0.5"),
        ("Output", "10 classes\nSoftmax"),
    ]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    box_width = 1.25
    box_height = 1.35
    start_x = 0.3
    y = 1.35
    gap = 0.25

    colors = [
        "#e8f0fe",
        "#dff5e1",
        "#fff2cc",
        "#dff5e1",
        "#fff2cc",
        "#f2e6ff",
        "#e6f7ff",
        "#ffe6e6",
        "#e8f0fe",
    ]

    for index, ((layer_name, layer_detail), color) in enumerate(zip(layers, colors)):
        x = start_x + index * (box_width + gap)
        rectangle = plt.Rectangle(
            (x, y),
            box_width,
            box_height,
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.4,
            joinstyle="round",
        )
        ax.add_patch(rectangle)

        ax.text(
            x + box_width / 2,
            y + 0.88,
            layer_name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x + box_width / 2,
            y + 0.42,
            layer_detail,
            ha="center",
            va="center",
            fontsize=8.5,
        )

        if index < len(layers) - 1:
            arrow_start = x + box_width
            arrow_end = x + box_width + gap
            ax.annotate(
                "",
                xy=(arrow_end, y + box_height / 2),
                xytext=(arrow_start, y + box_height / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#333333"),
            )

    ax.text(
        7,
        3.3,
        "CNN Architecture for MNIST Handwritten Digit Recognition",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"CNN architecture diagram saved to: {output_path}")


def main():
    generate_mnist_sample_grid()
    generate_cnn_architecture_diagram()


if __name__ == "__main__":
    main()
