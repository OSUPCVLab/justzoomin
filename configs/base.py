from pathlib import Path

from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()

    data_root = Path("/justzoomin_data")

    # --- Paths ---
    cfg.paths = ConfigDict()
    cfg.paths.metadata_csv = str(data_root / "metadata/large_area_train_map.csv")
    cfg.paths.ground_root = str(data_root / "streetview/images")
    cfg.paths.tile_layout = str(data_root / "satellite/layout.yaml")
    cfg.paths.log_dir = "./logs"
    cfg.paths.checkpoint_dir = "./checkpoints"

    # --- Data ---
    cfg.data = ConfigDict()
    cfg.data.grid_size = 4
    cfg.data.sequence_length = 4
    cfg.data.target_image_size = 384
    cfg.data.geographic_center_latlon = [38.8936, -77.0116]
    cfg.data.region_bounds_meters = [-3000.0, 7000.0, -5000.0, 5000.0]

    # --- Model Architecture ---
    cfg.model = ConfigDict()
    cfg.model.encoder_name = "facebook/dinov2-base"
    cfg.model.freeze_backbone = False
    cfg.model.decoder_num_heads = 8
    cfg.model.decoder_num_layers = 6

    # --- Training ---
    cfg.training = ConfigDict()
    cfg.training.batch_size = 64
    cfg.training.learning_rate = 3e-4
    cfg.training.num_epochs = 30
    cfg.training.log_interval = 20  # Log metrics every 20 steps
    cfg.training.eval_interval_epochs = 3   # Evaluate on the validation set every epoch
    cfg.training.save_interval_epochs = 3   # Save a checkpoint every 5 epochs
    cfg.training.compile = True             # Enable torch.compile for a speed boost
    cfg.training.grad_clip_norm = 1.0       # Max norm for gradient clipping. Set to 0 to disable.
    cfg.training.warmup_pct = 0.05          # Percentage of total steps for LR warmup
    cfg.training.label_smoothing = 0.1

    # --- Ground Data Augmentations ---
    cfg.aug = ConfigDict()
    cfg.aug.enable = True                    
    cfg.aug.ground_scale = [0.7, 1.0]        # RandomResizedCrop scale range
    cfg.aug.color_jitter = [0.2, 0.2, 0.2, 0.05]  # B, C, S, H
    cfg.aug.gray_p = 0.1
    cfg.aug.blur_p = 0.3
    cfg.aug.erase_p = 0.3                    
    cfg.aug.ground_cutting = 0   

    # --- Wandb Logging ---
    cfg.wandb = ConfigDict()
    cfg.wandb.enable = True                 # Set to False to disable logging
    cfg.wandb.project_name = "GeoZoom-CVGL"
        
    return cfg
