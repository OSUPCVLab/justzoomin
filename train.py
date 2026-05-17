import logging
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
import wandb
import numpy as np
import tiledwebmaps as twm
from PIL import Image
import torch.nn.functional as F

from configs.base import get_config
from configs.eval import get_config as get_eval_config
from data.dataset import ZoomDataset
from data.transforms import build_transforms
from models.model import GeoLocalizationModel
from utils.logger import setup_default_logging
from utils.utils import (
    DummyWandb, print_banner, generate_run_name, get_lr,
    compute_init, compute_cleanup, log0, dist_mean_scalar, seed_worker
)

# Configure logging
setup_default_logging()
logger = logging.getLogger('justzoomin')


# --------------------- Copied from evaluate_checkpoints.py ---------------------
class DynamicSequenceLoader:
    """
    Handles auto-regressive loading of satellite images based on model actions.
    """
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
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_evaluation(model, dataloader, criterion, device, sequence_length, ddp):
    """
    Runs a full auto-regressive evaluation pass (r@1, r@Xm) AND
    a teacher-forced validation loss pass in a single loop.
    """
    model_eval = model.module if ddp else model
    model_eval.eval()
    
    # Final-distance thresholds (meters) to report as "r@Xm"
    DISTANCE_THRESHOLDS_M = [40, 50, 100]

    dynamic_loader = DynamicSequenceLoader(dataloader.dataset, device)

    all_preds_ar, all_targets_ar = [], []
    all_gt_latlons, all_final_pred_centers = [], []
    
    total_loss_local = 0.0 # For teacher-forced loss
    
    dataloader.sampler.set_epoch(0) # Set epoch for sampler
    
    for batch in dataloader:
        # --- Data Setup ---
        ground_images = batch["ground"].to(device, non_blocking=True)
        satellite_sequence_tf = batch["satellite_sequence"].to(device, non_blocking=True)
        target_sequence_cpu = batch["sequence"] # (B, S), on CPU
        target_sequence_gpu = target_sequence_cpu.to(device, non_blocking=True) # For loss
        B = ground_images.shape[0]

        # --- 1. Teacher-Forced Loss Calculation ---
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits_tf = model_eval(
                ground_images=ground_images,
                satellite_sequence=satellite_sequence_tf,
                target_sequence=target_sequence_gpu
            )
            loss = criterion(logits_tf.view(-1, logits_tf.shape[-1]), target_sequence_gpu.view(-1))
        total_loss_local += loss.item()


        # --- 2. Auto-Regressive (AR) Evaluation ---
        gt_latlons_batch = np.stack([batch["meta"]["latitude"], batch["meta"]["longitude"]], axis=1)
        all_gt_latlons.append(gt_latlons_batch)

        # Encode ground once
        B_g, N_crops, C_g, H_g, W_g = ground_images.shape
        ground_images_flat = ground_images.view(B_g * N_crops, C_g, H_g, W_g)
        ground_global_flat, _ = model_eval.encoder._process_batch(ground_images_flat)
        ground_global = ground_global_flat.view(B_g, N_crops, -1).mean(dim=1)

        # Initialize AR state
        current_centers, current_sizes = dynamic_loader.get_initial_state(B)
        satellite_features_history = []
        action_history_tensor = torch.zeros((B, sequence_length), dtype=torch.long, device=device)
        step_predictions_ar = []

        for step in range(sequence_length):
            current_sat_images = dynamic_loader.get_images_for_state(current_centers, current_sizes)
            current_sat_features, _ = model_eval.encoder._process_batch(current_sat_images)
            satellite_features_history.append(current_sat_features)

            sat_feats_tensor = torch.stack(satellite_features_history, dim=1)
            sat_feats_padded = F.pad(sat_feats_tensor, (0, 0, 0, sequence_length - (step + 1)))

            logits_full_sequence = model_eval.decoder(
                ground_global_feature=ground_global,
                satellite_sequence_features=sat_feats_padded,
                target_sequence=action_history_tensor
            )
            logits_current_step = logits_full_sequence[:, step, :]
            predicted_action = torch.argmax(logits_current_step, dim=-1)  # (B,)

            step_predictions_ar.append(predicted_action)
            action_history_tensor[:, step] = predicted_action

            predicted_action_np = predicted_action.cpu().numpy()
            next_centers, next_sizes = dynamic_loader.update_state_for_actions(
                current_centers, current_sizes, predicted_action_np
            )

            if step == sequence_length - 1:
                all_final_pred_centers.append(next_centers)
            else:
                current_centers, current_sizes = next_centers, next_sizes

        all_preds_ar.append(torch.stack(step_predictions_ar, dim=1).cpu())
        all_targets_ar.append(target_sequence_cpu) # Use the CPU one


    # --- Calculate Metrics (Local) ---
    
    # 1. Loss Metric
    avg_loss_local = total_loss_local / len(dataloader)
    
    # 2. AR Metrics
    predictions = torch.cat(all_preds_ar)
    targets = torch.cat(all_targets_ar)
    gt_latlons_all = np.concatenate(all_gt_latlons)
    final_predicted_centers_all = np.concatenate(all_final_pred_centers)

    correct_steps = (predictions == targets)
    strict_correct_sequences = correct_steps.all(dim=1)
    r_at_1_strict_local = strict_correct_sequences.float().mean().item() * 100.0

    distances_meters = twm.geo.distance(final_predicted_centers_all, gt_latlons_all)
    
    metrics_local = {
        "val_loss": avg_loss_local,
        "val_r@1_strict": r_at_1_strict_local,
    }
    for thr in DISTANCE_THRESHOLDS_M:
        acc_local = (torch.from_numpy(distances_meters) <= thr).float().mean().item() * 100.0
        metrics_local[f"val_r@{thr}m"] = acc_local

    # --- Synchronize Metrics (DDP) ---
    if ddp:
        metrics_global = {}
        for k, v_local in metrics_local.items():
            metrics_global[k] = dist_mean_scalar(v_local)
    else:
        metrics_global = metrics_local
        
    return metrics_global


def main():
    # Configuration and Initialization
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init() 
    print_banner()
    cfg = get_config()
    eval_cfg = get_eval_config()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log0(f"Using device: {device}")
    
    global_batch_size = cfg.training.batch_size * ddp_world_size
    log0(f"Global batch size: {global_batch_size} ({cfg.training.batch_size} per GPU)")
    
    # Wandb Setup
    run_name = generate_run_name(cfg)
    if ddp_rank == 0:
        if cfg.wandb.enable:
            wandb.init(project=cfg.wandb.project_name, name=run_name, config=cfg.to_dict())
            tracker = wandb
            log0(f"Wandb run initialized: {run_name}")        
        else:
            tracker = DummyWandb()
    else:
        tracker = DummyWandb() # No logging on other ranks
        
    checkpoint_dir = Path(cfg.paths.checkpoint_dir) / run_name
    if ddp_rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Data Loading 
    train_transforms = build_transforms(cfg.data.target_image_size, aug_cfg=cfg.aug)
    train_dataset = ZoomDataset(cfg, transforms=train_transforms)
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size, # Per-GPU batch size
        sampler=train_sampler,              # Use sampler instead of shuffle=True
        num_workers=8, 
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=seed_worker         # Seed workers
    )

    val_transforms = build_transforms(eval_cfg.data.target_image_size, aug_cfg=getattr(eval_cfg, "aug", None))
    val_dataset = ZoomDataset(eval_cfg, transforms=val_transforms)
    val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size, # Per-GPU batch size
        sampler=val_sampler,                # Use sampler
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker         # Seed workers
    )
    log0(f"Loaded train dataset with {len(train_dataset)} samples.")
    log0(f"Loaded validation dataset with {len(val_dataset)} samples.")
    # Model, Optimizer, and Loss 
    model = GeoLocalizationModel(cfg)
    model.to(device)
    
    if cfg.training.compile:
        log0("Compiling the model... (this may take a minute)")        
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        log0("Model wrapped with DDP.") 
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')
    
    if cfg.wandb.enable:
        tracker.watch(model, log="all", log_freq=100)

    # Training Loop
    best_val_accuracy = 0.0
    total_steps = 0
    log0("Starting training...")

    # Calculate total steps for the learning rate scheduler
    max_training_steps = len(train_dataloader) * cfg.training.num_epochs
    warmup_steps = int(max_training_steps * cfg.training.warmup_pct)
    log0(f"Total training steps (per GPU): {max_training_steps}, Warmup steps: {warmup_steps}")
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}", disable=(ddp_rank != 0))

        for step, batch in enumerate(progress_bar):
            # Update learning rate at each step based on the scheduler
            lr = get_lr(total_steps, cfg.training.learning_rate, warmup_steps, max_training_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            ground_images = batch["ground"].to(device)
            satellite_sequence = batch["satellite_sequence"].to(device)
            target_sequence = batch["sequence"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(
                    ground_images=ground_images,
                    satellite_sequence=satellite_sequence,
                    target_sequence=target_sequence
                )
                loss = criterion(logits.view(-1, logits.shape[-1]), target_sequence.view(-1))
            
            # Scale loss and perform backward pass using the scaler
            scaler.scale(loss).backward()

            if cfg.training.grad_clip_norm > 0:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) 

            total_steps += 1
            loss_val = loss.item()
            
            if ddp_rank == 0:
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr:.2e}"})
            
            if total_steps % cfg.training.log_interval == 0:
                avg_loss_global_step = dist_mean_scalar(loss_val) # Average current step loss
                if ddp_rank == 0:
                    tracker.log({"train_loss": avg_loss_global_step, "train_lr": lr, "step": total_steps, "epoch": epoch + 1})
       
        # --- End of Epoch Evaluation and Checkpointing ---
        if (epoch + 1) % cfg.training.eval_interval_epochs == 0:
            val_metrics = run_evaluation(model, val_dataloader, criterion, device, cfg.data.sequence_length, ddp)
            if ddp_rank == 0:
                # Updated logging string to include Val Loss
                log_str = (
                    f"Epoch {epoch+1} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val r@1(strict): {val_metrics['val_r@1_strict']:.2f}% | "
                    f"Val r@40m: {val_metrics['val_r@40m']:.2f}%"
                )
                log0(log_str) 
                tracker.log(val_metrics)

                # Update best model logic to use the primary AR metric
                current_accuracy = val_metrics["val_r@40m"]
                if current_accuracy > best_val_accuracy:
                    best_val_accuracy = current_accuracy
                    best_checkpoint_path = checkpoint_dir / "best_model.pth"
                    model_state = model.module.state_dict() if ddp else model.state_dict()
                    torch.save({'model_state_dict': model_state}, best_checkpoint_path)
                    log0(f"Saved new best model with r@1(strict): {best_val_accuracy:.2f}%")


        if (epoch + 1) % cfg.training.save_interval_epochs == 0:
            if ddp_rank == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch+1}.pth"
                # Save the underlying model state dict
                model_state = model.module.state_dict() if ddp else model.state_dict()
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() # Save scaler state too
                }
                torch.save(save_dict, checkpoint_path)
                log0(f"Saved periodic checkpoint to {checkpoint_dir / checkpoint_path.name}")
        if ddp:
            dist.barrier()


    if ddp_rank == 0:
        tracker.finish()
    log0("Training finished.")

    # DDP Cleanup
    compute_cleanup()

if __name__ == "__main__":
    main()