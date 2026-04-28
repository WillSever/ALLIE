import csv
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from allie.data import rgb_float_to_uint8


def calculate_pair_metrics(pred_rgb, target_rgb):
    pred_uint8 = rgb_float_to_uint8(pred_rgb)
    target_uint8 = rgb_float_to_uint8(target_rgb)

    psnr_value = peak_signal_noise_ratio(
        target_uint8,
        pred_uint8,
        data_range=255,
    )
    ssim_value = structural_similarity(
        target_uint8,
        pred_uint8,
        channel_axis=-1,
        data_range=255,
    )
    return psnr_value, ssim_value


def summarize_metrics(rows):
    if not rows:
        return {"mean_psnr": np.nan, "mean_ssim": np.nan}

    return {
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
    }


def save_metrics_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["image", "target", "psnr", "ssim"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

