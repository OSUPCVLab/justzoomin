import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tiledwebmaps as twm

from configs.eval import get_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging
from utils.visualization_utils import (
    draw_grid_and_marker,
    load_full_satellite_path_with_details,
    generate_overview_visualization,
    tensor_to_image
)

# --------------------- User-configurable ---------------------
CHECKPOINT_DIR = Path("./checkpoints")
DEVICE = "cuda:0"
BATCH_SIZE = 64
NUM_SAMPLES_TO_VIZ = 10
OUTPUT_DIR = Path("./dataset_visualizations/evaluation_results_multi")

# Visualizations (off by default; same behavior as before)
GENERATE_DETAILED_VIZ = False
GENERATE_OVERVIEW_VIZ = False

# Final-distance thresholds (meters) to report as "r@Xm"
DISTANCE_THRESHOLDS_M = [40, 50, 100]
REPORT_FILENAME = "evaluation_report.txt"
# ------------------------------------------------------------

setup_default_logging()
logger = logging.getLogger(__name__)


class DynamicSequenceLoader:
    def __init__(self, dataset, device):
        self.tile_loader = dataset.tile_loader
        self.transform = dataset.transforms.get("satellite")
        if self.transform is None:
            raise ValueError("Satellite transforms are required for dynamic loading.")
        self.target_image_size = dataset.target_image_size
        self.grid_size = dataset.grid_size
        self.device = device
        self.initial_center_latlon = dataset.initial_center_latlon
        self.initial_image_size = dataset.initial_image_size

    def get_initial_state(self, batch_size):
        centers = np.tile(self.initial_center_latlon, (batch_size, 1))
        sizes = np.tile(self.initial_image_size, (batch_size))
        return centers, sizes

    def get_images_for_state(self, centers_latlon, sizes_meters):
        batch_images = []
        for center, size in zip(centers_latlon, sizes_meters):
            mpp = size / self.target_image_size
            array = self.tile_loader.load(
                latlon=center, bearing=0.0, meters_per_pixel=mpp,
                shape=(self.target_image_size, self.target_image_size)
            )
            image = Image.fromarray(array)
            batch_images.append(self.transform(image))
        return torch.stack(batch_images).to(self.device, non_blocking=True)

    def update_state_for_actions(self, centers_latlon, sizes_meters, actions_np):
        new_centers, new_sizes = [], []
        for center, size, action in zip(centers_latlon, sizes_meters, actions_np):
            patch_size = size / self.grid_size
            row, col = divmod(action, self.grid_size)
            offset_east = (col + 0.5) * patch_size - (size / 2.0)
            offset_north = -((row + 0.5) * patch_size - (size / 2.0))
            temp_center = twm.geo.move_from_latlon(center, 90, offset_east)
            next_center = twm.geo.move_from_latlon(temp_center, 0, offset_north)
            new_centers.append(next_center)
            new_sizes.append(patch_size)
        return np.array(new_centers), np.array(new_sizes)


def generate_detailed_visualization(sample, gt_sequence, pred_sequence, cfg, output_path):
    gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])
    gt_path_details, gt_final_patch = load_full_satellite_path_with_details(cfg, gt_sequence, gt_latlon)
    pred_path_details, pred_final_patch = load_full_satellite_path_with_details(cfg, pred_sequence, gt_latlon)

    ground_images = tensor_to_image(sample["ground"], denorm=True)
    if ground_images.ndim == 3:
        ground_images = np.expand_dims(ground_images, axis=0)
    num_ground_crops = ground_images.shape[0]

    total_cols = num_ground_crops + cfg.data.sequence_length + 1
    width_ratios = [1.5] * num_ground_crops + [1] * (cfg.data.sequence_length + 1)
    fig, axes = plt.subplots(2, total_cols, figsize=(3.5 * total_cols, 7), gridspec_kw={'width_ratios': width_ratios})

    axes[0, 0].set_ylabel("Ground Truth", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[0, i].imshow(ground_images[i]); axes[0, i].set_title(f"Query Crop {i+1}"); axes[0, i].axis('off')

    for j in range(cfg.data.sequence_length):
        ax = axes[0, num_ground_crops + j]
        details = gt_path_details[j]
        ax.imshow(details["image_raw"])
        ax.set_title(f"Step {j+1}\nGT Patch: {details['patch_index']}")
        draw_grid_and_marker(
            ax=ax, image_shape=details["image_raw"].shape, grid_size=cfg.data.grid_size,
            gt_pixel_coords=details["gt_pixels"],
            selected_patch_coords=divmod(details['patch_index'], cfg.data.grid_size),
        )

    ax_final_gt = axes[0, -1]
    ax_final_gt.imshow(gt_final_patch["image_raw"])
    ax_final_gt.set_title(f"Final View (Lvl {cfg.data.sequence_length})")
    ax_final_gt.plot(gt_final_patch["gt_pixels"][0], gt_final_patch["gt_pixels"][1], 'ro', markersize=6,
                     markeredgecolor='white', markeredgewidth=1.0)
    ax_final_gt.axis('off')

    axes[1, 0].set_ylabel("Predicted", fontweight='bold', size='large')
    for i in range(num_ground_crops):
        axes[1, i].imshow(ground_images[i]); axes[1, i].set_title(f"Query Crop {i+1}"); axes[1, i].axis('off')

    for j in range(cfg.data.sequence_length):
        ax = axes[1, num_ground_crops + j]
        details = pred_path_details[j]
        is_correct = (gt_sequence[j] == pred_sequence[j])
        ax.imshow(details["image_raw"])
        ax.set_title(f"Step {j+1}\nPred Patch: {details['patch_index']}")
        draw_grid_and_marker(
            ax=ax, image_shape=details["image_raw"].shape, grid_size=cfg.data.grid_size,
            gt_pixel_coords=details["gt_pixels"],
            selected_patch_coords=divmod(details["patch_index"], cfg.data.grid_size),
            is_correct=is_correct, draw_marker=False
        )

    ax_final_pred = axes[1, -1]
    ax_final_pred.imshow(pred_final_patch["image_raw"])
    ax_final_pred.set_title(f"Final View (Lvl {cfg.data.sequence_length})")
    ax_final_pred.axis('off')

    fig.suptitle(f"ID: {sample['meta']['image_id']} | GT: {gt_sequence} vs Pred: {pred_sequence}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, device: torch.device, cfg) -> dict:
    logger.info(f"========== Evaluating checkpoint: {checkpoint_path.name} ==========")

    # Model
    model = GeoLocalizationModel(cfg)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint['model_state_dict']
    clean_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # Data
    transforms = build_transforms(cfg.data.target_image_size)
    dataset = ZoomDataset(cfg, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Loaded dataset with {len(dataset)} samples for evaluation.")

    S = cfg.data.sequence_length
    if S != 4:
        logger.warning(f"R@k metrics are written assuming S=3. Found S={S}. r@1 (strict) may be incomparable across settings.")

    dynamic_loader = DynamicSequenceLoader(dataset, device)

    all_preds, all_targets = [], []
    all_gt_latlons, all_final_pred_centers = [], []

    for batch in tqdm(dataloader, desc="Auto-regressive evaluation"):
        ground_images = batch["ground"].to(device, non_blocking=True)
        target_sequence = batch["sequence"]  # (B, S)
        B = ground_images.shape[0]

        gt_latlons_batch = np.stack([batch["meta"]["latitude"], batch["meta"]["longitude"]], axis=1)
        all_gt_latlons.append(gt_latlons_batch)

        # Encode ground once
        B_g, N_crops, C_g, H_g, W_g = ground_images.shape
        ground_images_flat = ground_images.view(B_g * N_crops, C_g, H_g, W_g)
        ground_global_flat, _ = model.encoder._process_batch(ground_images_flat)
        ground_global = ground_global_flat.view(B_g, N_crops, -1).mean(dim=1)

        # Initialize AR state
        current_centers, current_sizes = dynamic_loader.get_initial_state(B)
        satellite_features_history = []
        action_history_tensor = torch.zeros((B, S), dtype=torch.long, device=device)
        step_predictions = []

        for step in range(S):
            current_sat_images = dynamic_loader.get_images_for_state(current_centers, current_sizes)
            current_sat_features, _ = model.encoder._process_batch(current_sat_images)
            satellite_features_history.append(current_sat_features)

            sat_feats_tensor = torch.stack(satellite_features_history, dim=1)
            sat_feats_padded = F.pad(sat_feats_tensor, (0, 0, 0, S - (step + 1)))

            logits_full_sequence = model.decoder(
                ground_global_feature=ground_global,
                satellite_sequence_features=sat_feats_padded,
                target_sequence=action_history_tensor
            )
            logits_current_step = logits_full_sequence[:, step, :]
            predicted_action = torch.argmax(logits_current_step, dim=-1)  # (B,)

            step_predictions.append(predicted_action)
            action_history_tensor[:, step] = predicted_action

            predicted_action_np = predicted_action.cpu().numpy()
            next_centers, next_sizes = dynamic_loader.update_state_for_actions(
                current_centers, current_sizes, predicted_action_np
            )

            if step == S - 1:
                all_final_pred_centers.append(next_centers)
            else:
                current_centers, current_sizes = next_centers, next_sizes

        all_preds.append(torch.stack(step_predictions, dim=1).cpu())
        all_targets.append(target_sequence)

    # Consolidate
    predictions = torch.cat(all_preds)          # (N, S)
    targets = torch.cat(all_targets)            # (N, S)
    gt_latlons_all = np.concatenate(all_gt_latlons)              # (N, 2)
    final_predicted_centers_all = np.concatenate(all_final_pred_centers)  # (N, 2)

    # Strict r@1 (exact patch sequence)
    correct_steps = (predictions == targets)
    strict_correct_sequences = correct_steps.all(dim=1)
    r_at_1_strict = strict_correct_sequences.float().mean().item() * 100.0

    # Distance-based @Xm using final predicted centers
    distances_meters = twm.geo.distance(final_predicted_centers_all, gt_latlons_all)  # (N,)
    metrics = {
        "checkpoint": checkpoint_path.name,
        "samples": int(distances_meters.shape[0]),
        "r@1_strict": r_at_1_strict,
    }
    for thr in DISTANCE_THRESHOLDS_M:
        acc = (torch.from_numpy(distances_meters) <= thr).float().mean().item() * 100.0
        metrics[f"r@{thr}m"] = acc

    # Optional sample visualizations (unchanged – still off by default)
    if (GENERATE_DETAILED_VIZ or GENERATE_OVERVIEW_VIZ) and len(dataset) > 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        num_samples = len(dataset)
        import numpy as _np
        indices = _np.random.choice(num_samples, min(NUM_SAMPLES_TO_VIZ, num_samples), replace=False)
        for idx in tqdm(indices, desc="Creating visualizations"):
            sample = dataset[idx]
            pred_seq = predictions[idx].numpy()
            gt_seq = targets[idx].numpy()
            if GENERATE_DETAILED_VIZ:
                out_p = OUTPUT_DIR / f"{checkpoint_path.stem}_detailed_{sample['meta']['image_id']}.png"
                generate_detailed_visualization(sample, gt_seq, pred_seq, cfg, out_p)
            if GENERATE_OVERVIEW_VIZ:
                gt_latlon = np.array([sample["meta"]["latitude"], sample["meta"]["longitude"]])
                out_p2 = OUTPUT_DIR / f"{checkpoint_path.stem}_overview_{sample['meta']['image_id']}.png"
                generate_overview_visualization(cfg, gt_seq, pred_seq, gt_latlon, sample['meta']['image_id'], out_p2)

    logger.info(f"Results for {checkpoint_path.name}: {metrics}")
    return metrics


def write_report(checkpoint_folder: Path, results: list):
    """
    Writes/updates a compact report file in the SAME checkpoint folder.
    One line per checkpoint for easy scanning.
    """
    report_path = checkpoint_folder / REPORT_FILENAME
    metric_columns = " | ".join([f"r@{thr}m(%)" for thr in DISTANCE_THRESHOLDS_M])
    header = (
        "# Evaluation Report\n"
        f"# Folder: {checkpoint_folder}\n"
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Columns: checkpoint | samples | r@1(strict %) | {metric_columns}\n\n"
    )
    lines = []
    for m in results:
        metric_values = "\t".join([f"{m.get(f'r@{thr}m', float('nan')):.2f}" for thr in DISTANCE_THRESHOLDS_M])
        line = f"{m['checkpoint']}\t{m['samples']}\t{m['r@1_strict']:.2f}\t{metric_values}\n"
        lines.append(line)

    # Write (overwrite each run for clarity)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header)
        for l in lines:
            f.write(l)
    logger.info(f"Saved report to: {report_path}")


@torch.no_grad()
def main():
    cfg = get_config()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not CHECKPOINT_DIR.exists() or not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    # Find checkpoints
    ckpts = sorted(CHECKPOINT_DIR.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {CHECKPOINT_DIR}")

    results = []
    for ckpt in ckpts:
        try:
            metrics = evaluate_checkpoint(ckpt, device, cfg)
        except Exception as e:
            logger.exception(f"Failed on {ckpt.name}: {e}")
            # Put a failed line in the report to keep track
            metrics = {"checkpoint": ckpt.name, "samples": 0, "r@1_strict": float('nan')}
            for thr in DISTANCE_THRESHOLDS_M:
                metrics[f"r@{thr}m"] = float('nan')
        results.append(metrics)

    # Save a single compact report in the same checkpoint folder
    write_report(CHECKPOINT_DIR, results)


if __name__ == "__main__":
    main()
