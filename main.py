"""
Main command-line entry point for the MNIST CNN project.

Examples:
    python main.py train
    python main.py predict-random --count 5
    python main.py predict-custom --image images/my_digit.png
    python main.py gui
"""

import argparse

import predict
import train


def main():
    parser = argparse.ArgumentParser(
        description="MNIST handwritten digit recognition using a CNN."
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train and evaluate the CNN.")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    train_parser.add_argument(
        "--batch-size", type=int, default=128, help="Training batch size."
    )
    train_parser.add_argument(
        "--model",
        default=str(train.DEFAULT_MODEL_PATH),
        help="Where to save the trained model.",
    )

    random_parser = subparsers.add_parser(
        "predict-random",
        help="Predict random unseen images from the MNIST test set.",
    )
    random_parser.add_argument("--count", type=int, default=5, help="Number of images.")
    random_parser.add_argument(
        "--model",
        default=str(predict.DEFAULT_MODEL_PATH),
        help="Path to the trained model.",
    )

    custom_parser = subparsers.add_parser(
        "predict-custom",
        help="Predict a local handwritten digit image.",
    )
    custom_parser.add_argument(
        "--image",
        required=True,
        help="Path to the custom image, for example images/my_digit.png.",
    )
    custom_parser.add_argument(
        "--model",
        default=str(predict.DEFAULT_MODEL_PATH),
        help="Path to the trained model.",
    )

    subparsers.add_parser("gui", help="Open the optional digit drawing app.")

    args = parser.parse_args()

    if args.command == "train":
        train.train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_path=args.model,
        )
    elif args.command == "predict-random":
        predict.predict_random_test_images(count=args.count, model_path=args.model)
    elif args.command == "predict-custom":
        predict.predict_custom_image(image_path=args.image, model_path=args.model)
    elif args.command == "gui":
        import gui

        gui.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
