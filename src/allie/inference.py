from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from allie.data import load_rgb_image, save_rgb_image
from allie.metrics import calculate_pair_metrics, save_metrics_csv, summarize_metrics
from allie.utils import ensure_dir, list_image_files


def load_trained_model(model_path):
    return load_model(str(model_path), compile=False)


def predict_image(model, image_path, width, height):
    image = load_rgb_image(image_path, width, height)
    batch = np.expand_dims(image, axis=0)
    prediction = model.predict(batch, verbose=0)
    return image, np.squeeze(prediction, axis=0)


def run_inference(
    model,
    input_dir,
    predictions_dir,
    width,
    height,
    target_dir=None,
    comparisons_dir=None,
    metrics_path=None,
    save_predictions=True,
    save_comparisons=True,
):
    input_files = list_image_files(input_dir)
    target_files = list_image_files(target_dir) if target_dir else []

    if target_dir and len(input_files) != len(target_files):
        raise ValueError(
            "Input and target folders have different numbers of images: "
            f"{len(input_files)} vs {len(target_files)}"
        )

    ensure_dir(predictions_dir)
    if comparisons_dir:
        ensure_dir(comparisons_dir)

    rows = []

    for index, input_path in enumerate(input_files):
        target_path = target_files[index] if target_files else None
        input_rgb, pred_rgb = predict_image(model, input_path, width, height)

        if save_predictions:
            save_rgb_image(pred_rgb, Path(predictions_dir) / input_path.name)

        if target_path:
            target_rgb = load_rgb_image(target_path, width, height)
            psnr_value, ssim_value = calculate_pair_metrics(pred_rgb, target_rgb)
            rows.append(
                {
                    "image": input_path.name,
                    "target": target_path.name,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                }
            )

            if save_comparisons and comparisons_dir:
                comparison = np.concatenate([input_rgb, pred_rgb, target_rgb], axis=1)
                save_rgb_image(comparison, Path(comparisons_dir) / input_path.name)

            print(
                f"{input_path.name} | PSNR: {psnr_value:.4f} | SSIM: {ssim_value:.4f}"
            )
        else:
            print(f"{input_path.name} | prediction saved")

    if rows and metrics_path:
        save_metrics_csv(rows, metrics_path)
        summary = summarize_metrics(rows)
        print("=" * 32)
        print(f"Mean PSNR: {summary['mean_psnr']:.4f}")
        print(f"Mean SSIM: {summary['mean_ssim']:.4f}")
        print(f"Metrics saved at: {metrics_path}")

    return rows

