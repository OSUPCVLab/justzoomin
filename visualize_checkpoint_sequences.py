import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs.eval import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from evaluate_checkpoints import DynamicSequenceLoader
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging
from utils.visualization_utils import (
    draw_grid_and_marker,
    generate_overview_visualization,
    load_full_satellite_path_with_details,
    tensor_to_image,
)

# --------------------- User-configurable ---------------------
CHECKPOINT_PATH = Path("./checkpoints/best_model.pth")
OUTPUT_DIR = Path("./dataset_visualizations/checkpoint_sequences")
DEVICE = "cuda:0"

NUM_SAMPLES = 10
RANDOM_SEED = 42

GENERATE_SEQUENCE_VIZ = True
GENERATE_OVERVIEW_VIZ = True
# ------------------------------------------------------------

setup_default_logging()
logger = logging.getLogger(__name__)


def clean_state_dict(state_dict):
    clean = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        clean[key] = value
    return clean


def load_model(cfg, checkpoint_path, device):
    model = GeoLocalizationModel(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_sequence(model, dynamic_loader, sample, cfg, device):
    ground_images = sample["ground"].unsqueeze(0).to(device, non_blocking=True)
    B, num_crops, C, H, W = ground_images.shape

    ground_images_flat = ground_images.view(B * num_crops, C, H, W)
    ground_global_flat, _ = model.encoder._process_batch(ground_images_flat)
    ground_global = ground_global_flat.view(B, num_crops, -1).mean(dim=1)

    current_centers, current_sizes = dynamic_loader.get_initial_state(B)
    satellite_features_history = []
    action_history_tensor = torch.zeros((B, cfg.data.sequence_length), dtype=torch.long, device=device)
    predictions = []

    for step in range(cfg.data.sequence_length):
        current_sat_images = dynamic_loader.get_images_for_state(current_centers, current_sizes)
        current_sat_features, _ = model.encoder._process_batch(current_sat_images)
        satellite_features_history.append(current_sat_features)

        sat_feats_tensor = torch.stack(satellite_features_history, dim=1)
        sat_feats_padded = F.pad(sat_feats_tensor, (0, 0, 0, cfg.data.sequence_length - (step + 1)))

        logits = model.decoder(
            ground_global_feature=ground_global,
            satellite_sequence_features=sat_feats_padded,
            target_sequence=action_history_tensor,
        )
        action = torch.argmax(logits[:, step, :], dim=-1)
        predictions.append(int(action.item()))
        action_history_tensor[:, step] = action

        current_centers, current_sizes = dynamic_loader.update_state_for_actions(
            current_centers, current_sizes, action.cpu().numpy()
        )

    return np.array(predictions, dtype=np.int64)


def generate_sequence_visualization(sample, gt_sequence, pred_sequence, cfg, output_path):
    gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])
    gt_path_details, gt_final_patch = load_full_satellite_path_with_details(cfg, gt_sequence, gt_latlon)
    pred_path_details, pred_final_patch = load_full_satellite_path_with_details(cfg, pred_sequence, gt_latlon)

    ground_images = tensor_to_image(sample["ground"], denorm=True)
    if ground_images.ndim == 3:
        ground_images = np.expand_dims(ground_images, axis=0)

    num_ground_crops = ground_images.shape[0]
    total_cols = num_ground_crops + cfg.data.sequence_length + 1
    width_ratios = [1.5] * num_ground_crops + [1] * (cfg.data.sequence_length + 1)
    fig, axes = plt.subplots(
        2,
        total_cols,
        figsize=(3.4 * total_cols, 7),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    axes[0, 0].set_ylabel("Ground Truth", fontweight="bold", size="large")
    axes[1, 0].set_ylabel("Predicted", fontweight="bold", size="large")

    for row in range(2):
        for i in range(num_ground_crops):
            axes[row, i].imshow(ground_images[i])
            axes[row, i].set_title(f"Query Crop {i + 1}")
            axes[row, i].axis("off")

    for step in range(cfg.data.sequence_length):
        gt_details = gt_path_details[step]
        gt_ax = axes[0, num_ground_crops + step]
        gt_ax.imshow(gt_details["image_raw"])
        gt_ax.set_title(f"Step {step + 1}\nGT: {gt_details['patch_index']}")
        draw_grid_and_marker(
            ax=gt_ax,
            image_shape=gt_details["image_raw"].shape,
            grid_size=cfg.data.grid_size,
            gt_pixel_coords=gt_details["gt_pixels"],
            selected_patch_coords=divmod(gt_details["patch_index"], cfg.data.grid_size),
        )

        pred_details = pred_path_details[step]
        pred_ax = axes[1, num_ground_crops + step]
        pred_ax.imshow(pred_details["image_raw"])
        pred_ax.set_title(f"Step {step + 1}\nPred: {pred_details['patch_index']}")
        draw_grid_and_marker(
            ax=pred_ax,
            image_shape=pred_details["image_raw"].shape,
            grid_size=cfg.data.grid_size,
            gt_pixel_coords=pred_details["gt_pixels"],
            selected_patch_coords=divmod(pred_details["patch_index"], cfg.data.grid_size),
            is_correct=gt_sequence[step] == pred_sequence[step],
            draw_marker=False,
        )

    axes[0, -1].imshow(gt_final_patch["image_raw"])
    axes[0, -1].set_title("Final GT View")
    axes[0, -1].plot(
        gt_final_patch["gt_pixels"][0],
        gt_final_patch["gt_pixels"][1],
        "ro",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    axes[0, -1].axis("off")

    axes[1, -1].imshow(pred_final_patch["image_raw"])
    axes[1, -1].set_title("Final Pred View")
    axes[1, -1].axis("off")

    image_id = sample["meta"]["image_id"]
    fig.suptitle(f"ID: {image_id} | GT: {gt_sequence.tolist()} | Pred: {pred_sequence.tolist()}", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    cfg = get_config()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")

    model = load_model(cfg, CHECKPOINT_PATH, device)
    transforms = build_transforms(cfg.data.target_image_size, aug_cfg=getattr(cfg, "aug", None))
    dataset = ZoomDataset(cfg, transforms=transforms)
    dynamic_loader = DynamicSequenceLoader(dataset, device)

    rng = np.random.default_rng(RANDOM_SEED)
    num_samples = min(NUM_SAMPLES, len(dataset))
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    for idx in tqdm(indices, desc="Visualizing checkpoint sequences"):
        sample = dataset[int(idx)]
        gt_sequence = sample["sequence"].numpy()
        pred_sequence = predict_sequence(model, dynamic_loader, sample, cfg, device)
        image_id = sample["meta"]["image_id"]
        gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])

        if GENERATE_SEQUENCE_VIZ:
            output_path = OUTPUT_DIR / f"{CHECKPOINT_PATH.stem}_{image_id}_sequence.png"
            generate_sequence_visualization(sample, gt_sequence, pred_sequence, cfg, output_path)

        if GENERATE_OVERVIEW_VIZ:
            output_path = OUTPUT_DIR / f"{CHECKPOINT_PATH.stem}_{image_id}_overview.png"
            generate_overview_visualization(cfg, gt_sequence, pred_sequence, gt_latlon, image_id, output_path)

    logger.info(f"Saved visualizations to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
