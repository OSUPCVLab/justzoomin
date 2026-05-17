from pathlib import Path

from ml_collections import ConfigDict

def get_config():
    cfg = ConfigDict()

    data_root = Path("/justzoomin_data")

    # --- Paths ---
    cfg.paths = ConfigDict()
    cfg.paths.metadata_csv = str(data_root / "metadata/large_area_val_map.csv")
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
    cfg.model.freeze_backbone = True
    cfg.model.decoder_num_heads = 8
    cfg.model.decoder_num_layers = 6
    
    # --- Ground Data Augmentations ---
    cfg.aug = ConfigDict()
    cfg.aug.enable = False                    
    cfg.aug.ground_scale = [0.7, 1.0]        # RandomResizedCrop scale range
    cfg.aug.color_jitter = [0.2, 0.2, 0.2, 0.05]  # B, C, S, H
    cfg.aug.gray_p = 0.1
    cfg.aug.blur_p = 0.2
    cfg.aug.erase_p = 0.1  
    cfg.aug.ground_cutting = 0   

    # --- Training ---
    cfg.training = ConfigDict()
    cfg.training.batch_size = 128
    cfg.training.learning_rate = 3e-4
    cfg.training.num_epochs = 50
    cfg.training.log_interval = 20  # Log metrics every 20 steps
    cfg.training.project_name = "GeoZoom-CVGL"
    cfg.training.run_name = "dinov2-base-run-1" # Give a name for this specific run

    return cfg
