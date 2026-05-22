# Report Screenshot Guide

Use this checklist to collect all pictures for section **15. Output Pictures**.

## Before You Start

Open a terminal inside the project folder:

```bash
cd "c:\Users\pc\Desktop\New folder\MNIST_CNN_Project"
```

Create and activate the virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If TensorFlow does not install with your current Python, install a
TensorFlow-compatible Python version and recreate the virtual environment.

## 1. MNIST Dataset Sample Image

Generate the image:

```bash
python report_assets.py
```

Use this file in your report:

```text
images/mnist_samples_0_to_9.png
```

This image shows one sample digit from 0 to 9.

## 2. CNN Architecture Diagram

The same command also creates the architecture diagram:

```bash
python report_assets.py
```

Use this file:

```text
images/cnn_architecture_diagram.png
```

It shows:

```text
Input -> Convolution -> Pooling -> Convolution -> Pooling -> Flatten -> Dense -> Output
```

## 3. Training Accuracy/Loss Graph

Train the model:

```bash
python train.py --epochs 5
```

Use this generated file:

```text
images/training_accuracy_loss.png
```

## 4. Confusion Matrix

The training command also creates:

```text
images/confusion_matrix.png
```

Use this image in the evaluation/results section.

## 5. Random Prediction Output

After training, run:

```bash
python predict.py --random --count 5
```

Use this generated file:

```text
images/random_predictions.png
```

It shows random unseen test images with actual and predicted labels.

## 6. Custom Digit Prediction Screenshot

1. Write one digit on paper or in Paint.
2. Save it as:

```text
images/my_digit.png
```

3. Run:

```bash
python predict.py --image images/my_digit.png
```

4. Use either:

```text
images/custom_prediction.png
```

or take a screenshot of the prediction window.

On Windows, press `Win + Shift + S`, drag over the image window, then paste the
screenshot into your report.

## 7. GUI Screenshot

Open the drawing app:

```bash
python gui.py
```

Draw a digit on the canvas and click **Predict**.

Take a screenshot with `Win + Shift + S` showing:

- The drawing canvas
- The predicted digit
- The confidence value

Save or paste that screenshot into your report.

## 8. Terminal Screenshot

Run training again or scroll to the training output:

```bash
python train.py --epochs 5
```

Take a screenshot of the terminal area that shows:

- Training accuracy
- Validation accuracy
- Test loss
- Test accuracy

Good lines to capture:

```text
Test Loss: ...
Test Accuracy: ...
Model saved to: ...
```

On Windows, use `Win + Shift + S`, select the terminal area, and paste the
screenshot into your report.

## Final Image Checklist

Use these files/screenshots in the report:

```text
images/mnist_samples_0_to_9.png
images/cnn_architecture_diagram.png
images/training_accuracy_loss.png
images/confusion_matrix.png
images/random_predictions.png
images/custom_prediction.png
GUI screenshot from python gui.py
Terminal screenshot from python train.py --epochs 5
```
