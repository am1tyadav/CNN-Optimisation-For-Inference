import tensorflow as tf
import numpy as np
from numpy.typing import DTypeLike
from tf_keras import models
from time import perf_counter
from model import get_data


def convert_to_tflite(model: models.Model, model_filepath: str, quantise: bool = False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantise:
        x_train, _, _, _ = get_data()
        x_train = x_train.astype(np.float32)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # int8 quantization
        def representative_data_gen():
            for input_value in (
                tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100)
            ):
                yield [input_value]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(model_filepath, "wb") as f:
        f.write(tflite_model)


def speed_test(tflite_model_filepath: str, dtype: DTypeLike):
    iterations = 1000

    interpreter = tf.lite.Interpreter(model_path=tflite_model_filepath)
    interpreter.allocate_tensors()

    dummy_input = np.zeros((1, 32, 32, 3), dtype=dtype)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    start_time = perf_counter()

    for _ in range(0, iterations):
        interpreter.set_tensor(input_index, dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)

    end_time = perf_counter()
    time_per_iteration = (end_time - start_time) / iterations

    print(f"For model: {tflite_model_filepath}")
    print(f"Time per iteration: {time_per_iteration:.6f} seconds")
