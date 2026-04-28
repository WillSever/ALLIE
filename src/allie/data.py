from pathlib import Path

import cv2
import numpy as np

from allie.utils import list_image_files


def load_rgb_image(image_path, width, height, normalize=True):
    """Load an image with OpenCV, convert BGR to RGB, resize, and normalize."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if normalize:
        image = image.astype(np.float32) / 255.0

    return image


def load_image_pairs(input_dir, target_dir, width, height):
    """Load sorted Low/Normal image pairs into memory, matching the notebook."""
    input_files = list_image_files(input_dir)
    target_files = list_image_files(target_dir)

    if len(input_files) != len(target_files):
        raise ValueError(
            "Input and target folders have different numbers of images: "
            f"{len(input_files)} vs {len(target_files)}"
        )

    inputs = []
    targets = []
    pairs = []

    for input_path, target_path in zip(input_files, target_files):
        inputs.append(load_rgb_image(input_path, width, height))
        targets.append(load_rgb_image(target_path, width, height))
        pairs.append((input_path.name, target_path.name))

    return np.array(inputs), np.array(targets), pairs


def rgb_float_to_uint8(image):
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def save_rgb_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = rgb_float_to_uint8(image)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

