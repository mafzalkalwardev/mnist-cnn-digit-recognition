"""
Optional Tkinter drawing app for testing the trained MNIST CNN model.

Run:
    python gui.py

Train the model first:
    python train.py
"""

from pathlib import Path
from tkinter import BOTH, BOTTOM, Button, Canvas, Label, LEFT, Tk, messagebox

import numpy as np
from PIL import Image, ImageDraw

from predict import DEFAULT_MODEL_PATH, prepare_pil_image
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
DRAWING_SIZE = 280
BRUSH_SIZE = 18


class DigitDrawingApp:
    """A simple drawing canvas for real-time digit prediction."""

    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Drawing App")
        self.root.resizable(False, False)

        self.model = None
        self.last_x = None
        self.last_y = None

        self.canvas = Canvas(
            root,
            width=DRAWING_SIZE,
            height=DRAWING_SIZE,
            bg="black",
            cursor="cross",
        )
        self.canvas.pack(fill=BOTH, expand=True)

        self.result_label = Label(root, text="Draw a digit, then click Predict.")
        self.result_label.pack(pady=8)

        predict_button = Button(root, text="Predict", command=self.predict_digit)
        predict_button.pack(side=LEFT, padx=12, pady=10)

        clear_button = Button(root, text="Clear", command=self.clear_canvas)
        clear_button.pack(side=LEFT, padx=12, pady=10)

        close_button = Button(root, text="Close", command=root.destroy)
        close_button.pack(side=BOTTOM, pady=8)

        self.image = Image.new("L", (DRAWING_SIZE, DRAWING_SIZE), color=0)
        self.image_draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.last_x is None or self.last_y is None:
            self.start_drawing(event)
            return

        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle="round",
            smooth=True,
        )
        self.image_draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=255,
            width=BRUSH_SIZE,
        )

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, _event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (DRAWING_SIZE, DRAWING_SIZE), color=0)
        self.image_draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit, then click Predict.")

    def load_model_if_needed(self):
        if self.model is not None:
            return True

        if not DEFAULT_MODEL_PATH.exists():
            messagebox.showerror(
                "Model Not Found",
                f"Train the model first:\npython train.py\n\nMissing: {DEFAULT_MODEL_PATH}",
            )
            return False

        self.model = load_model(DEFAULT_MODEL_PATH)
        return True

    def predict_digit(self):
        if not self.load_model_if_needed():
            return

        model_input, _display_image = prepare_pil_image(self.image)
        prediction = self.model.predict(model_input, verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        self.result_label.config(
            text=f"Prediction: {predicted_digit}    Confidence: {confidence * 100:.2f}%"
        )


def main():
    root = Tk()
    DigitDrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
