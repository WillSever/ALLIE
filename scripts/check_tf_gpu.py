import os
from _windows_dll_paths import configure_windows_dll_paths

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
configure_windows_dll_paths()

import tensorflow as tf


def main():
    print(f"TensorFlow: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")

    print("Devices:")
    for device in devices:
        print(f"  - {device}")

    if gpus:
        print("GPU is available.")
    else:
        print("GPU was not detected.")


if __name__ == "__main__":
    main()
