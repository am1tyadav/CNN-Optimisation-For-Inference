import tensorflow as tf
from tf_keras import layers, models, datasets


def get_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


class RegularCNNBlock(models.Model):
    def __init__(self, out_channels: int, kernel_size: int, name: str):
        super().__init__(name=name)

        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def fuse_conv_bn_weights(
    conv: layers.Conv2D, bn: layers.BatchNormalization
) -> tuple[tf.Tensor]:
    """This function fuses the weights of a convolution and batch norm layer.
    This does not need to be a tf.function as it is only called once so
    the graph, if built, will never be re-used.
    """
    filters = conv.kernel
    bias = conv.bias if conv.use_bias else tf.zeros(conv.filters)

    # Get batch norm parameters
    gamma = bn.gamma
    beta = bn.beta
    mean = bn.moving_mean
    var = bn.moving_variance
    epsilon = bn.epsilon

    # Fuse parameters and return the new weights
    scale = gamma / tf.sqrt(var + epsilon)
    fused_filters = filters * scale[None, None, None, :]
    fused_bias = (bias - mean) * scale + beta

    return fused_filters, fused_bias


class FusedCNNBlock(models.Model):
    def __init__(self, out_channels: int, kernel_size: int, name: str):
        super().__init__(name=name)

        self.conv = layers.Conv2D(
            out_channels, kernel_size, padding="same", activation="relu"
        )

    def update_conv_weights(self, regular_block: RegularCNNBlock):
        fused_filters, fused_bias = fuse_conv_bn_weights(
            regular_block.conv, regular_block.bn
        )

        self.conv.kernel.assign(fused_filters)
        self.conv.bias.assign(fused_bias)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.conv(x)
        return x


def create_model(
    num_classes: int = 10,
    input_shape: tuple[int] = (32, 32, 3),
    is_fused: bool = False,
    channels: tuple[int] = (32, 64, 128),
    filters: tuple[int] = (3, 3, 3),
    pool_size: tuple[int] = (2, 2, 2),
) -> models.Model:
    inputs = layers.Input(shape=input_shape, name="input")

    x = inputs
    block_count = 0

    cnn_block = FusedCNNBlock if is_fused else RegularCNNBlock

    for c, f, p in zip(channels, filters, pool_size):
        block_count += 1

        block = cnn_block(c, f, name=f"block_{block_count}")
        x = block(x)
        x = layers.MaxPooling2D(p, name=f"pool_{block_count}")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, x, name="classifier")
    return model
