'''
some of it taken from 
https://github.com/karpathy/nanochat/blob/master/nanochat/common.py
https://github.com/huggingface/nanoVLM/tree/main
'''

import os
import math
import time
import torch
import random
import numpy as np
import torch.distributed as dist

import logging
from .logger import setup_default_logging

setup_default_logging()
logger = logging.getLogger(__name__)

BANNER = r"""
       █████                     █████       ███████████                                               ███            
      ░░███                     ░░███       ░█░░░░░░███                                               ░░░             
       ░███  █████ ████  █████  ███████     ░     ███░    ██████   ██████  █████████████              ████  ████████  
       ░███ ░░███ ░███  ███░░  ░░░███░           ███     ███░░███ ███░░███░░███░░███░░███  ██████████░░███ ░░███░░███ 
       ░███  ░███ ░███ ░░█████   ░███           ███     ░███ ░███░███ ░███ ░███ ░███ ░███ ░░░░░░░░░░  ░███  ░███ ░███ 
 ███   ░███  ░███ ░███  ░░░░███  ░███ ███     ████     █░███ ░███░███ ░███ ░███ ░███ ░███             ░███  ░███ ░███ 
░░████████   ░░████████ ██████   ░░█████     ███████████░░██████ ░░██████  █████░███ █████            █████ ████ █████
 ░░░░░░░░     ░░░░░░░░ ░░░░░░     ░░░░░     ░░░░░░░░░░░  ░░░░░░   ░░░░░░  ░░░░░ ░░░ ░░░░░            ░░░░░ ░░░░ ░░░░░ 
                        the Ohio State University -  Photogrammetric Computer Vision Lab - 2025                                
"""

def seed_worker(worker_id):
    """Sets random seed for dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_run_name(cfg):
    """Generates a descriptive run name from the config."""
    # Model details
    encoder_name = cfg.model.encoder_name.split('/')[-1]
    decoder_details = f"decL{cfg.model.decoder_num_layers}H{cfg.model.decoder_num_heads}"

    # Training details
    batch_size = f"bs{cfg.training.batch_size}"
    learning_rate = f"lr{cfg.training.learning_rate}"
    epochs = f"e{cfg.training.num_epochs}"

    # Timestamp
    date = time.strftime("%m%d-%H%M")

    # Combine everything
    run_name = (
        f"{encoder_name}_{decoder_details}_"
        f"{batch_size}_{learning_rate}_{epochs}_{date}"
    )
    return run_name

def print_banner():
    """Prints the project banner to the console."""
    print0(BANNER)
    
def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
        
def log0(msg, level=logging.INFO, *args, **kwargs):
    """Logs a message only on the main process (rank 0)."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        if level == logging.DEBUG:
            logger.debug(msg, *args, **kwargs)
        elif level == logging.INFO:
            logger.info(msg, *args, **kwargs)
        elif level == logging.WARNING:
            logger.warning(msg, *args, **kwargs)
        elif level == logging.ERROR:
            logger.error(msg, *args, **kwargs)
        elif level == logging.CRITICAL:
            logger.critical(msg, *args, **kwargs)
        else: # Default to INFO if level is invalid
             logger.info(msg, *args, **kwargs)

def is_ddp():
    """Checks if the script is running in distributed mode."""
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    """Gets DDP rank, local rank, and world size."""
    if is_ddp():
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
    
def compute_init():
    """Initializes DDP environment and sets device."""
    ddp = 'RANK' in os.environ # Check if RANK env var is set
    ddp_rank = int(os.environ.get('RANK', 0))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
    ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)

    # Set seed based on rank for better randomness across processes
    seed = 42 + ddp_rank 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Precision
    torch.set_float32_matmul_precision("high")

    if ddp:
        dist.init_process_group(backend="nccl")
        print0(f"DDP initialized: world_size={ddp_world_size}, rank={ddp_rank}, local_rank={ddp_local_rank}")
        dist.barrier() # Sync all processes

    print0(f"Process {ddp_rank} using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Cleans up the DDP process group."""
    if is_ddp():
        dist.destroy_process_group()

def dist_mean_scalar(value):
    """Averages a scalar value across all DDP processes."""
    if not is_ddp():
        return value
    
    tensor = torch.tensor(value, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / dist.get_world_size()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
    def watch(self, *args, **kwargs):
        pass

def get_lr(it, max_lr, warmup_steps, max_steps):
    """Calculates the learning rate for a given step."""
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > max_steps, return min learning rate
    if it > max_steps:
        return max_lr * 0.1  # min_lr is 10% of max_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (max_lr * 0.1) + coeff * (max_lr * 0.9)