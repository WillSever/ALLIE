import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError


mse_fn = MeanSquaredError()


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def psnr_loss(y_true, y_pred):
    return -tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def combined_loss_factory(mse_weight=0.73, ssim_weight=0.27):
    """Create the ALLIE loss: weighted MSE + weighted SSIM loss."""

    def combined_loss(y_true, y_pred):
        mse = mse_fn(y_true, y_pred)
        ssim_l = ssim_loss(y_true, y_pred)
        return mse_weight * mse + ssim_weight * ssim_l

    combined_loss.__name__ = "combined_loss"
    return combined_loss

