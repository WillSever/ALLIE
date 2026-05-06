import tensorflow as tf
from tensorflow.keras import Input, layers, models
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from allie.losses_tf import combined_loss_factory, ssim_loss


def self_attention_module(x):
    """Self-attention block used in the ALLIE bottleneck."""
    num_filters = x.shape[-1]
    attention = layers.Conv2D(num_filters, kernel_size=1, padding="same")(x)
    attention = layers.BatchNormalization()(attention)
    attention = layers.Activation("relu")(attention)
    attention = layers.Conv2D(num_filters, kernel_size=1, padding="same")(attention)
    attention = layers.Activation("sigmoid")(attention)
    return layers.Multiply()([x, attention])


def match_shape(to_adjust, target):
    """Adjust one tensor spatially to match another using crop or padding."""
    h_diff = to_adjust.shape[1] - target.shape[1]
    w_diff = to_adjust.shape[2] - target.shape[2]

    if h_diff > 0 or w_diff > 0:
        crop_top = max(h_diff // 2, 0)
        crop_bottom = max(h_diff - crop_top, 0)
        crop_left = max(w_diff // 2, 0)
        crop_right = max(w_diff - crop_left, 0)
        to_adjust = layers.Cropping2D(
            ((crop_top, crop_bottom), (crop_left, crop_right))
        )(to_adjust)

    if h_diff < 0 or w_diff < 0:
        pad_top = max((-h_diff) // 2, 0)
        pad_bottom = max((-h_diff) - pad_top, 0)
        pad_left = max((-w_diff) // 2, 0)
        pad_right = max((-w_diff) - pad_left, 0)
        to_adjust = layers.ZeroPadding2D(
            ((pad_top, pad_bottom), (pad_left, pad_right))
        )(to_adjust)

    return to_adjust


def build_allie_model(
    input_shape,
    encoder_filters,
    bottleneck_filters,
    decoder_filters,
    dropout_rate,
    learning_rate,
    kernel_size=3,
    mse_weight=0.73,
    ssim_weight=0.27,
):
    """Build the paper-base ALLIE model with four encoder blocks."""
    if len(encoder_filters) != 4:
        raise ValueError("paper_base_4_blocks expects exactly 4 encoder filters.")
    if len(decoder_filters) != 4:
        raise ValueError("paper_base_4_blocks expects exactly 4 decoder filters.")

    inputs = Input(shape=input_shape)

    def encoder_block(x, filters):
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        pooled = layers.MaxPooling2D((2, 2))(x)
        return x, pooled

    b = layers.Conv2D(
        bottleneck_filters,
        (kernel_size, kernel_size),
        padding="same",
    )

    def decoder_block(x, skip, filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = match_shape(x, skip)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(
            filters,
            (kernel_size, kernel_size),
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    x1, p1 = encoder_block(inputs, encoder_filters[0])
    x2, p2 = encoder_block(p1, encoder_filters[1])
    x3, p3 = encoder_block(p2, encoder_filters[2])
    x4, p4 = encoder_block(p3, encoder_filters[3])

    b = b(p4)
    b = layers.BatchNormalization()(b)
    b = layers.Activation("relu")(b)
    b = self_attention_module(b)

    u1 = decoder_block(b, x4, decoder_filters[0])
    u2 = decoder_block(u1, x3, decoder_filters[1])
    u3 = decoder_block(u2, x2, decoder_filters[2])
    u4 = decoder_block(u3, x1, decoder_filters[3])

    outputs = layers.Conv2D(
        3,
        (kernel_size, kernel_size),
        activation="sigmoid",
        padding="same",
    )(u4)

    model = models.Model(inputs=inputs, outputs=outputs, name="ALLIE")
    model.compile(
        optimizer=Adam(learning_rate),
        loss=combined_loss_factory(mse_weight, ssim_weight),
        metrics=[
            MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanMetricWrapper(ssim_loss, name="ssim_loss"),
        ],
    )
    return model
