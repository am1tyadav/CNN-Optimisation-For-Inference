import click
import numpy as np
from tf_keras import models
from train import create_and_train_model, create_fused_model, create_model
from tflite import convert_to_tflite, speed_test, evaluate
from enum import Enum


class Filepaths(str, Enum):
    CNN = "data/regular_cnn.keras"
    FUSED_CNN = "data/fused_cnn.keras"
    CNN_TFLITE = "data/regular_cnn.tflite"
    FUSED_CNN_TFLITE = "data/fused_cnn.tflite"
    CNN_INT8 = "data/regular_cnn_int8.tflite"
    FUSED_CNN_INT8 = "data/fused_cnn_int8.tflite"


def load_models() -> tuple[models.Model, models.Model]:
    cnn = create_model(is_fused=False)
    cnn.load_weights(Filepaths.CNN)
    fused_cnn = create_model(is_fused=True)
    fused_cnn.load_weights(Filepaths.FUSED_CNN)
    return cnn, fused_cnn


@click.group()
def cli():
    click.echo("CNN training and optimisation CLI")


@cli.command(help="Train CNN, save and convert to Fused CNN")
def train():
    cnn = create_and_train_model()
    fused_cnn = create_fused_model(cnn)

    cnn.save_weights(Filepaths.CNN)
    fused_cnn.save_weights(Filepaths.FUSED_CNN)


@cli.command(help="Convert to fp32 and int8 TFLite models for both CNN and Fused CNN")
def tflite():
    cnn, fused_cnn = load_models()

    convert_to_tflite(cnn, Filepaths.CNN_TFLITE)
    convert_to_tflite(fused_cnn, Filepaths.FUSED_CNN_TFLITE)
    convert_to_tflite(cnn, Filepaths.CNN_INT8, quantise=True)
    convert_to_tflite(fused_cnn, Filepaths.FUSED_CNN_INT8, quantise=True)


@cli.command(help="Run speed test on all TFLite models")
def speed():
    speed_test(Filepaths.CNN_TFLITE, np.float32)
    speed_test(Filepaths.FUSED_CNN_TFLITE, np.float32)
    speed_test(Filepaths.CNN_INT8, np.uint8)
    speed_test(Filepaths.FUSED_CNN_INT8, np.uint8)


@cli.command(help="Evaluate all TFLite models")
def eval():
    evaluate(Filepaths.CNN_TFLITE, np.float32)
    evaluate(Filepaths.FUSED_CNN_TFLITE, np.float32)
    evaluate(Filepaths.CNN_INT8, np.uint8)
    evaluate(Filepaths.FUSED_CNN_INT8, np.uint8)


if __name__ == "__main__":
    cli()
