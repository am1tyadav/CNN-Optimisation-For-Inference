import numpy as np
from tf_keras import models, callbacks
from model import create_model, get_data
from time import perf_counter


def evaluate_model(model: models.Model, name: str):
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    _, _, x_test, y_test = get_data()

    start_time = perf_counter()
    val_loss, val_acc = model.evaluate(x_test, y_test)
    end_time = perf_counter()

    print(f"======================= {name} =======================")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Number of parameters: {model.count_params()}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")


def create_and_train_model(epochs: int = 10) -> models.Model:
    batch_size = 32
    x_train, y_train, x_test, y_test = get_data()
    model = create_model()

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )

    evaluate_model(model, "Regular CNN")
    return model


def create_fused_model(
    trained_model: models.Model,
    num_classes: int = 10,
    input_shape: tuple[int] = (32, 32, 3),
) -> models.Model:
    fused_model = create_model(num_classes, input_shape, is_fused=True)
    dummy_input = np.zeros((1, *input_shape), dtype=np.float32)
    _ = fused_model(dummy_input)  # To ensure the model is built

    for fused_layer, regular_layer in zip(
        fused_model.layers[1:], trained_model.layers[1:]
    ):
        if regular_layer.name.startswith("block"):
            fused_layer.update_conv_weights(regular_layer)
        else:
            fused_layer.set_weights(regular_layer.get_weights())

    fused_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    evaluate_model(fused_model, "Fused CNN")
    return fused_model
