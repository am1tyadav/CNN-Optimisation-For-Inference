import click
import numpy as np
from tf_keras import models
from train import (
    create_and_train_model,
    create_fused_model,
    create_model,
    evaluate_model,
)
from tflite import convert_to_tflite, speed_test, evaluate
from enum import Enum


class Filepaths(str, Enum):
    CNN = "data/regular_cnn.keras"
    FUSED_CNN = "data/fused_cnn.keras"
    SEPARABLE_CNN = "data/separable_cnn.keras"
    CNN_TFLITE = "data/regular_cnn.tflite"
    FUSED_CNN_TFLITE = "data/fused_cnn.tflite"
    SEPARABLE_TFLITE = "data/separable_cnn.tflite"
    CNN_INT8 = "data/regular_cnn_int8.tflite"
    FUSED_CNN_INT8 = "data/fused_cnn_int8.tflite"
    SEPARABLE_INT8 = "data/separable_cnn_int8.tflite"


def load_models() -> tuple[models.Model, models.Model]:
    cnn = create_model(is_fused=False)
    cnn.load_weights(Filepaths.CNN)
    fused_cnn = create_model(is_fused=True)
    fused_cnn.load_weights(Filepaths.FUSED_CNN)
    separable_cnn = create_model(is_separable=True)
    separable_cnn.load_weights(Filepaths.SEPARABLE_CNN)
    return cnn, fused_cnn, separable_cnn


@click.group()
def cli():
    click.echo("CNN training and optimisation CLI")


@cli.command(help="Train CNN, save and convert to Fused CNN")
def train():
    cnn = create_and_train_model()
    fused_cnn = create_fused_model(cnn)

    cnn.save_weights(Filepaths.CNN)
    fused_cnn.save_weights(Filepaths.FUSED_CNN)

    separable_cnn = create_and_train_model(epochs=20, is_separable=True)
    separable_cnn.save_weights(Filepaths.SEPARABLE_CNN)


@cli.command(help="Convert to fp32 and int8 TFLite models for both CNN and Fused CNN")
def tflite():
    cnn, fused_cnn, separable_cnn = load_models()

    convert_to_tflite(cnn, Filepaths.CNN_TFLITE)
    convert_to_tflite(fused_cnn, Filepaths.FUSED_CNN_TFLITE)
    convert_to_tflite(separable_cnn, Filepaths.SEPARABLE_TFLITE)
    convert_to_tflite(cnn, Filepaths.CNN_INT8, quantise=True)
    convert_to_tflite(fused_cnn, Filepaths.FUSED_CNN_INT8, quantise=True)
    convert_to_tflite(separable_cnn, Filepaths.SEPARABLE_INT8, quantise=True)


@cli.command(help="Run speed test on all TFLite models")
def speed():
    speed_test(Filepaths.CNN_TFLITE, np.float32)
    speed_test(Filepaths.FUSED_CNN_TFLITE, np.float32)
    speed_test(Filepaths.SEPARABLE_TFLITE, np.float32)
    speed_test(Filepaths.CNN_INT8, np.uint8)
    speed_test(Filepaths.FUSED_CNN_INT8, np.uint8)
    speed_test(Filepaths.SEPARABLE_INT8, np.uint8)


@cli.command(help="Evaluate all models")
def eval():
    cnn, fused_cnn, separable_cnn = load_models()

    evaluate_model(cnn, "CNN")
    evaluate_model(fused_cnn, "Fused CNN")
    evaluate_model(separable_cnn, "Separable CNN")

    evaluate(Filepaths.CNN_TFLITE, np.float32)
    evaluate(Filepaths.FUSED_CNN_TFLITE, np.float32)
    evaluate(Filepaths.SEPARABLE_TFLITE, np.float32)
    evaluate(Filepaths.CNN_INT8, np.uint8)
    evaluate(Filepaths.FUSED_CNN_INT8, np.uint8)
    evaluate(Filepaths.SEPARABLE_INT8, np.uint8)


@cli.command(help="Display the number of parameters in each model")
def params():
    cnn, fused_cnn, separable_cnn = load_models()

    print(f"CNN: {cnn.count_params()}")
    print(f"Fused CNN: {fused_cnn.count_params()}")
    print(f"Separable CNN: {separable_cnn.count_params()}")


if __name__ == "__main__":
    cli()
