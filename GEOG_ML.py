import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import glob
import random
import tifffile as tiff
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import sys

from time import time
from pdb import set_trace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Hyperparameter Config
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 150
WARMUP_EPOCHS = 15
PATCH_SIZE = 128
VIT_PATCH_SIZE = 4
MASK_RATIO = 0.7
SAMPLES_PER_FILE = 500
DROPOUT = 0.1
GRADIENT_LOSS_WEIGHT = 0.15 #.15
RELU_PENALTY_WEIGHT = 0.1
EARLY_STOP_PATIENCE = 25
# Tries to grab GPU if you have it. Had some trouble with this at first. Needed to download a a specific verison of torch to match
# my GPU. On my 4070 this code took ~16 hours to run, so heads up
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ViT Config
# Residual (1) + DOY (1) + SNODAS (4) + Static (12) + Cloud Mask (1) + Valid Mask (1) = 20 Channels
IN_CHANS = 20
OUT_CHANS = 1
EMBED_DIM = 256
DECODER_EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
NUM_DECODER_LAYERS = 6

#########################################################################################################################################
###### CHANGE THESE TO WHATEVER LOCAL DIR YOU DOWNLOAD FILES TO ######
BASE_PATH = Path("/home/rtheurer/repos/classproject/geospatial_data")
DATA_DIR        = BASE_PATH / "dynamic_days"
STATIC_DIR      = BASE_PATH / "static_topo"
BASE_OUTPUT_DIR = BASE_PATH / "training_outputs"
# DON"T CHANGE THIS EMPTY ONE!
OUTPUT_DIR = ''

# ENCODER SELECTION: CAN CHOOSE APE, alibi_2d, alibi_3d
# POSITIONAL_EMBEDDING_LIST = ['APE', 'alibi_2d', 'alibi_3d'] change to this to run all embeddings
POSITIONAL_EMBEDDING_LIST = ['APE']
#########################################################################################################################################

# Date time string to make output folder and save output with
now = datetime.now()
DATE_TIME = now.strftime("%Y%m%d_%H%M")

# Static Filenames: found in the STATIC_DIR
FN_DTM       = 'CA_acqMB_20240922_dtm_r1330_full_extent_fixed_2.tif'
FN_CURV_GEN  = 'CA_acqMB_20240922_dtm_r1330_full_extent_general_curvature_fixed_2.tif'
FN_CURV_PLAN = 'CA_acqMB_20240922_dtm_r1330_full_extent_plan_curvature_fixed_2.tif'
FN_TPI_9     = 'CA_acqMB_20240922_dtm_r1330_full_extent_tpi_9_fixed_2.tif'
FN_TPI_101   = 'CA_acqMB_20240922_dtm_r1330_full_extent_tpi_101_fixed_2.tif'
FN_SLOPE     = 'CA_acqMB_20240922_dtmslpaspmos_r1330_full_extent_slope_fixed_2.tif'
FN_EAST      = 'CA_acqMB_20240922_dtmslpaspmos_r1330_full_extent_eastness_fixed_2.tif'
FN_NORTH     = 'CA_acqMB_20240922_dtmslpaspmos_r1330_full_extent_northness_fixed_2.tif'
FN_R         = 'CA_acqMB_20240922_specrgb_r956_full_extent_r_fixed_2.tif'
FN_G         = 'CA_acqMB_20240922_specrgb_r956_full_extent_g_fixed_2.tif'
FN_B         = 'CA_acqMB_20240922_specrgb_r956_full_extent_b_fixed_2.tif'
FN_CANOPY    = 'CA_acqMB_20240922_canopyhgt_r1330_full_extent_fixed_2.tif'

STATIC_FILENAMES = [
    FN_DTM, FN_CURV_GEN, FN_CURV_PLAN, FN_TPI_9, FN_TPI_101,
    FN_SLOPE, FN_EAST, FN_NORTH,
    FN_R, FN_G, FN_B, FN_CANOPY
]

def load_static_images():
    """Load static terrain TIFs once. Returns list of 12 numpy arrays."""
    imgs = []
    for fn in STATIC_FILENAMES:
        path = os.path.join(STATIC_DIR, fn)
        img = tiff.imread(path).astype(np.float32)
        img[img == -9999] = 0.0
        img[np.isnan(img)] = 0.0
        imgs.append(img)
    return imgs

def verify_data_integrity(folder_list):
    '''Checks to see that all necessary files are available'''
    print("\n*** Verifying Data Integrity ***")
    valid_count = 0
    for i, folder in enumerate(folder_list):
        if i >= 5: break 
        #res_files = glob.glob(os.path.join(folder, "*residual*") )
        res_files = glob.glob(os.path.join(folder, "*ASO*") )
        doy_files = glob.glob(os.path.join(folder, "*acqdoy*") )
        snodas_files = glob.glob(os.path.join(folder, "*SNODAS*") )
        
        if res_files and doy_files and len(snodas_files) == 4:
            valid_count += 1
    
    if valid_count == 0:
        print("\nERROR: No valid folders found!")
        sys.exit(1)
    print("*** Data Check Passed ***\n")

'''Code for running APE'''
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 
    pos = pos.reshape(-1) 
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 
    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) 
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = np.concatenate([
        get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]), 
        get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 
    ], axis=1)
    return pos_embed

def get_2d_terrain_bias(elevation_patches, grid_size, num_heads):
    '''Calculates 2D ALiBi bias'''
    B, N = elevation_patches.shape
    device = elevation_patches.device
    
    coords_h = torch.arange(grid_size, device=device).float()
    coords_w = torch.arange(grid_size, device=device).float()
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    
    grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) 
    
    # Compute xy distance matrix
    dist_xy = torch.cdist(grid, grid, p=2) 
    dist_2d = dist_xy.unsqueeze(0).expand(B, -1, -1)
    
    # Create ALiBi slopes 
    start = 2 ** (-2 ** -3)
    ratio = start
    slopes = [start * ratio ** i for i in range(num_heads)]
    slopes = torch.tensor(slopes, device=device).view(1, num_heads, 1, 1)
    
    # Apply bias penalty 
    bias = -1.0 * slopes * dist_2d.unsqueeze(1)
    bias = bias.reshape(B * num_heads, N, N)
    
    return bias

def get_3d_terrain_bias(elevation_patches, grid_size, num_heads):
    '''Calculates 3D ALiBi bias'''
    B, N = elevation_patches.shape
    device = elevation_patches.device
    
    coords_h = torch.arange(grid_size, device=device).float()
    coords_w = torch.arange(grid_size, device=device).float()
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    
    grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) 
    
    # Compute xy distance matrix
    dist_xy = torch.cdist(grid, grid, p=2) 
    
    # Calculate z distance (elevation)
    elev_i = elevation_patches.unsqueeze(2) 
    elev_j = elevation_patches.unsqueeze(1) 
    dist_z = torch.abs(elev_i - elev_j)
    
    # Combine into 3D distance
    dist_3d = torch.sqrt(dist_xy.unsqueeze(0)**2 + (dist_z)**2)
    
    # Create ALiBi slopes 
    start = 2 ** (-2 ** -3)
    ratio = start
    slopes = [start * ratio ** i for i in range(num_heads)]
    slopes = torch.tensor(slopes, device=device).view(1, num_heads, 1, 1)
    
    # Apply bias penalty
    bias = -1.0 * slopes * dist_3d.unsqueeze(1)
    bias = bias.view(B * num_heads, N, N)
    
    return bias

def get_6d_terrain_bias(elev_patches, slope_patches, aspect_e_patches, aspect_n_patches, 
                        grid_size, num_heads):
    '''Experiment I tried for 6D ALiBi, didn't work too well. Feel free to disregard this'''
    B, N = elev_patches.shape
    device = elev_patches.device
    
    # Compute xy distance matrix
    coords_h = torch.arange(grid_size, device=device).float()
    coords_w = torch.arange(grid_size, device=device).float()
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1) 
    dist_xy = torch.cdist(grid, grid, p=2).unsqueeze(0) # [1, N, N]
    
    # Calculate z distance (elevation)
    dist_z = torch.abs(elev_patches.unsqueeze(2) - elev_patches.unsqueeze(1))
    
    # # Calculate s distance (slope)
    dist_s = torch.abs(slope_patches.unsqueeze(2) - slope_patches.unsqueeze(1))
    
    # Calculate e,n distance (distance of East/North)
    dist_e = aspect_e_patches.unsqueeze(2) - aspect_e_patches.unsqueeze(1)
    dist_n = aspect_n_patches.unsqueeze(2) - aspect_n_patches.unsqueeze(1)
    
    # Combine into 6D Distance
    term_xy = dist_xy**2
    term_z = (dist_z * .5)**2
    term_s = (dist_s * .5)**2
    term_e = (dist_e * .5)**2
    term_n = (dist_n * .5)**2
    
    dist_6d = torch.sqrt(term_xy + term_z + term_s + term_e + term_n)
    
    # Create ALiBi slopes 
    start = 2 ** (-2 ** -3)
    ratio = start
    slopes = [start * ratio ** i for i in range(num_heads)]
    slopes = torch.tensor(slopes, device=device).view(1, num_heads, 1, 1)
    
    # Apply bias penalty
    bias = -1.0 * slopes * dist_6d.unsqueeze(1)
    bias = bias.reshape(B * num_heads, N, N)
    
    return bias

def compute_dataset_stats(folder_list):
    # Compute normalization stats for input data
    print("Computing dataset statistics (Mean/Std)...")

    static_filenames = [
        FN_DTM, FN_CURV_GEN, FN_CURV_PLAN, FN_TPI_9, FN_TPI_101,
        FN_SLOPE, FN_EAST, FN_NORTH,
        FN_R, FN_G, FN_B,FN_CANOPY
    ] 

    static_stack = []
    for fn in static_filenames:
        path = os.path.join(STATIC_DIR, fn) 
        img = tiff.imread(path).flatten().astype(np.float32)
        img[img == -9999] = np.nan
        static_stack.append(img)
    static_stack = np.stack(static_stack, axis=1)

    # Random sample from folders
    sample_folders = random.sample(folder_list, min(len(folder_list), 20))
    all_pixels = []

    print(f"Sampling {len(sample_folders)} folders for statistics...")

    for folder_path in sample_folders:
        try:
            #res_files = glob.glob(os.path.join(folder_path, "*residual*") )
            res_files = glob.glob(os.path.join(folder_path, "*ASO*") )
            doy_files = glob.glob(os.path.join(folder_path, "*acqdoy*") )
            # Sort SNODAS data in chronological order
            snodas_files = sorted(glob.glob(os.path.join(folder_path, "*SNODAS*") ))

            if not res_files or not doy_files or len(snodas_files) != 4: 
                print(f"Skipping {os.path.basename(folder_path)}: Missing files.")
                continue

            # Set -9999 fill as NaN
            res = tiff.imread(res_files[0]).flatten().astype(np.float32)
            doy = tiff.imread(doy_files[0]).flatten().astype(np.float32)
            snodas = [tiff.imread(f).flatten().astype(np.float32) for f in snodas_files]            
            res[res == -9999] = np.nan
            doy[doy == -9999] = np.nan
            for s in snodas: s[s == -9999] = np.nan

            dynamic_stack = np.stack([res, doy] + snodas, axis=1)

            if len(res) != len(static_stack):
                print(f"Skipping {os.path.basename(folder_path)}: Size mismatch.")
                continue

            full_stack = np.concatenate([dynamic_stack, static_stack], axis=1)

            indices = np.random.choice(full_stack.shape[0], size=full_stack.shape[0]//100, replace=False)
            all_pixels.append(full_stack[indices])

        except Exception as e:
            print(f"Error reading {os.path.basename(folder_path)}: {e}")
            continue

    if not all_pixels:
        raise RuntimeError("Could not collect stats from data.")

    big_stack = np.concatenate(all_pixels, axis=0) 
    
    mean = np.nanmean(big_stack, axis=0)
    std = np.nanstd(big_stack, axis=0)
    
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std == 0] = 1.0

    print(f"Stats Computed. Mean: {mean[0]:.2f}, Std: {std[0]:.2f}")
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}

def masked_mse_loss(pred, target, valid_mask):
    # Calulate MSE loss
    squared_error = (pred - target) ** 2
    masked_error = squared_error * valid_mask
    loss = masked_error.sum() / (valid_mask.sum() + 1e-6)
    return loss

def gradient_loss(pred, target, valid_mask):
    # Horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    # Vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    # Mask
    mask_dx = valid_mask[:, :, :, :-1] * valid_mask[:, :, :, 1:]
    mask_dy = valid_mask[:, :, :-1, :] * valid_mask[:, :, 1:, :]
    # Masked gradient error
    dx_error = ((pred_dx - target_dx) ** 2) * mask_dx
    dy_error = ((pred_dy - target_dy) ** 2) * mask_dy
    loss_dx = dx_error.sum() / (mask_dx.sum() + 1e-6)
    loss_dy = dy_error.sum() / (mask_dy.sum() + 1e-6)
    return (loss_dx + loss_dy) / 2.0

def laplacian_loss(pred, target, valid_mask):
    # Define the 3x3 discrete Laplacian kernel
    kernel = torch.tensor([[[[0.0,  1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0,  1.0, 0.0]]]], device=pred.device)
    
    # Apply the kernel using 2D convolution
    # Padding=1 ensures the output shape matches the input shape
    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)
    
    # Calculate the masked Mean Squared Error of the Laplacians
    squared_error = (pred_lap - target_lap) ** 2
    masked_error = squared_error * valid_mask
    
    loss = masked_error.sum() / (valid_mask.sum() + 1e-6)
    return loss

class GeoFolderDataset(Dataset):

    def __init__(self, folder_list, stats, static_imgs, patch_size=128, augment=False, samples_per_file=1):
        self.folder_list = folder_list
        self.stats = stats
        self.patch_size = patch_size
        self.augment = augment
        self.samples_per_file = samples_per_file

        self.mean = self.stats['mean'][:, None, None]
        self.std = self.stats['std'][:, None, None]

        self.static_imgs = static_imgs

        self._cache = {}
        print(f"Caching {len(folder_list)} folders into RAM...", end=" ", flush=True)
        for folder_path in folder_list:
            res_files = glob.glob(os.path.join(folder_path, "*ASO*"))
            doy_files = glob.glob(os.path.join(folder_path, "*acqdoy*"))
            snodas_files = sorted(glob.glob(os.path.join(folder_path, "*SNODAS*")))

            if not res_files or not doy_files or len(snodas_files) != 4:
                continue

            res_img = tiff.imread(res_files[0]).astype(np.float32)
            doy_img = tiff.imread(doy_files[0]).astype(np.float32)
            snodas_imgs = [tiff.imread(f).astype(np.float32) for f in snodas_files]

            valid_mask = ((res_img != -9999) & ~np.isnan(res_img)).astype(np.float32)

            res_img[res_img == -9999] = 0.0
            res_img[np.isnan(res_img)] = 0.0
            doy_img[doy_img == -9999] = 0.0
            doy_img[np.isnan(doy_img)] = 0.0

            for i in range(len(snodas_imgs)):
                snodas_imgs[i][snodas_imgs[i] == -9999] = 0.0
                snodas_imgs[i][np.isnan(snodas_imgs[i])] = 0.0

            all_channels = [res_img, doy_img] + snodas_imgs + self.static_imgs
            image_stack = np.stack(all_channels, axis=0).astype(np.float32)

            image_stack = (image_stack - self.mean) / self.std
            image_stack = np.nan_to_num(image_stack, nan=0.0, posinf=0.0, neginf=0.0)

            self._cache[folder_path] = (image_stack, valid_mask)
        print(f"done ({len(self._cache)} folders cached)")

    def __len__(self):
        return len(self.folder_list) * self.samples_per_file

    def __getitem__(self, idx):
        folder_idx = idx // self.samples_per_file
        if folder_idx >= len(self.folder_list): folder_idx = 0
        folder_path = self.folder_list[folder_idx]

        image_stack, valid_mask = self._cache[folder_path]
        c, h, w = image_stack.shape

        for _ in range(10):
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            vm_patch = valid_mask[top:top+self.patch_size, left:left+self.patch_size]
            if vm_patch.mean() >= 0.2:
                break

        patch = image_stack[:, top:top+self.patch_size, left:left+self.patch_size].copy()

        tensor_stack = torch.from_numpy(patch)

        # Augment training data, don't augment validation data
        # Can only add random noise can't flip or rotate because aspect matters
        if self.augment:
            noise = torch.randn_like(tensor_stack) * 0.02
            tensor_stack = tensor_stack + noise

        valid_mask_patch = torch.from_numpy(
            valid_mask[top:top+self.patch_size, left:left+self.patch_size].copy()
        ).unsqueeze(0)

        return tensor_stack, valid_mask_patch

def generate_multiscale_mask(batch_size, size=128, target_ratio=0.75):
    masks = torch.ones(batch_size, 1, size, size, device=DEVICE)
    total_pixels = size * size

    for i in range(batch_size):
        masked_pixels = 0
        while masked_pixels / total_pixels < target_ratio:
            mode = random.randint(0, 3)
            if mode == 0: min_s, max_s = 0.02, 0.05
            elif mode == 1: min_s, max_s = 0.05, 0.20
            elif mode == 2: min_s, max_s = 0.20, 0.50
            else: min_s, max_s = 0.02, 0.40

            cloud_w = random.randint(max(1, int(size * min_s)), int(size * max_s))
            cloud_h = random.randint(max(1, int(size * min_s)), int(size * max_s))

            x = random.randint(0, size - cloud_w)
            y = random.randint(0, size - cloud_h)

            masks[i, 0, y:y+cloud_h, x:x+cloud_w] = 0.0
            masked_pixels = total_pixels - masks[i, 0].sum().item()

    return masks


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(identity + out)

class RefinementHead(nn.Module):
    # 3-block refinement CNN to smooth patch boundaries
    def __init__(self, out_chans, hidden_dim=64):
        super().__init__()
        self.proj_in = nn.Conv2d(out_chans, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )
        self.proj_out = nn.Conv2d(hidden_dim, out_chans, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.proj_in(x)
        out = self.blocks(out)
        out = self.proj_out(out)
        # skip connecton
        return identity + out

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, pos_emb_selected, img_size=128, patch_size=16, in_chans=20,
                 out_chans=1, embed_dim=256, depth=6, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=8,
                 mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.pos_emb_selected = pos_emb_selected
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if pos_emb_selected == 'APE':
            pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
            self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        elif pos_emb_selected in ['alibi_2d', 'alibi_3d', 'alibi_6d']:
            self.register_slopes_and_grid_for_alibi(num_heads)
        else:
            raise RuntimeError(f'User selected position embedding does not match any correct option: {pos_emb_selected}')

        # ENCODER LAYER
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim*mlp_ratio),
                                                   activation='gelu', batch_first=True, norm_first=True,
                                                   dropout=dropout)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, self.grid_size)
        self.register_buffer('decoder_pos_embed', torch.from_numpy(decoder_pos_embed).float().unsqueeze(0), persistent=False)
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_num_heads,
                                                   dim_feedforward=int(decoder_embed_dim*mlp_ratio),
                                                   activation='gelu', batch_first=True, norm_first=True,
                                                   dropout=dropout)
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        # Decoder predicts snow depth from channel 1
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * out_chans, bias=True)

        self.refinement_head = RefinementHead(out_chans, hidden_dim=64)

        # random weights
        self.initialize_weights()

    def register_slopes_and_grid_for_alibi(self, num_heads):
        start = 2 ** (-2 ** -3)
        ratio = start
        slopes = [start * ratio ** i for i in range(num_heads)]
        slopes_tensor = torch.tensor(slopes).view(1, num_heads, 1, 1)

        self.register_buffer('alibi_slopes', slopes_tensor)

        coords_h = torch.arange(self.grid_size)
        coords_w = torch.arange(self.grid_size)
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
        grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        self.register_buffer('grid_coords', grid)

    def initialize_weights(self):
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def patchify(self, imgs):
        p = self.patch_size
        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        c = self.out_chans
        h = w = int(x.shape[1]**.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def get_attn_bias_2d(self, x):
        # Get elev data fir alibi bias
        elev_map = x[:, 6:7, :, :]

        elev_patches = torch.nn.functional.adaptive_avg_pool2d(elev_map, (self.grid_size, self.grid_size))
        elev_patches = elev_patches.flatten(2).squeeze(1)

        attn_bias = get_2d_terrain_bias(
            elev_patches, 
            self.grid_size, 
            num_heads=8, 
        )     
        return attn_bias

    def get_attn_bias_3d(self, x):
        # Get elev data fir alibi bias
        elev_map = x[:, 6:7, :, :] 
      
        elev_patches = torch.nn.functional.adaptive_avg_pool2d(elev_map, (self.grid_size, self.grid_size))
        elev_patches = elev_patches.flatten(2).squeeze(1) 
        
        attn_bias = get_3d_terrain_bias(
            elev_patches, 
            self.grid_size, 
            num_heads=8, 
        )     
        return attn_bias
    
    def get_attn_bias_6d(self, x):
        # Attention bias retrieval code that didn't work, please ignore
        elev_map = x[:, 6:7, :, :]
        slope_map = x[:, 11:12, :, :]
        aspect_e_map = x[:, 12:13, :, :]
        aspect_n_map = x[:, 13:14, :, :]
      
        pool = torch.nn.functional.adaptive_avg_pool2d
        elev_patches = pool(elev_map, (self.grid_size, self.grid_size)).flatten(2).squeeze(1)
        slope_patches = pool(slope_map, (self.grid_size, self.grid_size)).flatten(2).squeeze(1)
        aspect_e_patches = pool(aspect_e_map, (self.grid_size, self.grid_size)).flatten(2).squeeze(1)
        aspect_n_patches = pool(aspect_n_map, (self.grid_size, self.grid_size)).flatten(2).squeeze(1)
        
        attn_bias = get_6d_terrain_bias(
            elev_patches, slope_patches, aspect_e_patches, aspect_n_patches,
            self.grid_size, num_heads=8
        )
        return attn_bias

    def forward_encoder(self, x):
        if self.pos_emb_selected == 'APE':
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed
            x = self.blocks(x)
        elif self.pos_emb_selected == 'alibi_2d':
            attn_bias = self.get_attn_bias_2d(x)
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.blocks(x, mask=attn_bias)
        elif self.pos_emb_selected == 'alibi_3d':
            attn_bias = self.get_attn_bias_3d(x)
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.blocks(x, mask=attn_bias)
        elif self.pos_emb_selected == 'alibi_6d':
            # IGNORE!
            attn_bias = self.get_attn_bias_6d(x)
            x = self.patch_embed(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.blocks(x, mask=attn_bias)
        else:
            raise RuntimeError(f'User selected position embedding does not match any correct option: {self.pos_emb_selected}')
        
        # Normalize data
        x = self.norm(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        # Unpatchify into [B, 1, H, W] snow depth image
        coarse_img = self.unpatchify(pred)
        # Smooth patch boundaries with the refinement CNN
        fine_img = self.refinement_head(coarse_img)

        # Return image [B, 1, H, W]
        return fine_img

# Writes all hyperparameters to the output text file for record keeping
def write_hyperparameters_to_file(epoch_output_file, positional_embedding):
    epoch_output_file.write(f"BATCH_SIZE={BATCH_SIZE}\n")
    epoch_output_file.write(f"LR={LR}\n")
    epoch_output_file.write(f"EPOCHS={EPOCHS}\n")
    epoch_output_file.write(f"WARMUP_EPOCHS={WARMUP_EPOCHS}\n")
    epoch_output_file.write(f"PATCH_SIZE={PATCH_SIZE}\n")
    epoch_output_file.write(f"VIT_PATCH_SIZE={VIT_PATCH_SIZE}\n")
    epoch_output_file.write(f"MASK_RATIO={MASK_RATIO}\n")
    epoch_output_file.write(f"SAMPLES_PER_FILE={SAMPLES_PER_FILE}\n")
    epoch_output_file.write(f"DROPOUT={DROPOUT}\n")
    epoch_output_file.write(f"GRADIENT_LOSS_WEIGHT={GRADIENT_LOSS_WEIGHT}\n")
    epoch_output_file.write(f"RELU_PENALTY_WEIGHT={RELU_PENALTY_WEIGHT}\n")
    epoch_output_file.write(f"EARLY_STOP_PATIENCE={EARLY_STOP_PATIENCE}\n")
    epoch_output_file.write(f"IN_CHANS={IN_CHANS}\n")
    epoch_output_file.write(f"OUT_CHANS={OUT_CHANS}\n")
    epoch_output_file.write(f"EMBED_DIM={EMBED_DIM}\n")
    epoch_output_file.write(f"DECODER_EMBED_DIM={DECODER_EMBED_DIM}\n")
    epoch_output_file.write(f"NUM_HEADS={NUM_HEADS}\n")
    epoch_output_file.write(f"NUM_LAYERS={NUM_LAYERS}\n")
    epoch_output_file.write(f"NUM_DECODER_LAYERS={NUM_DECODER_LAYERS}\n")
    epoch_output_file.write(f"POSITIONAL_EMBEDDING={positional_embedding}\n")
    epoch_output_file.write(f"DEVICE={DEVICE}\n")
    epoch_output_file.write("\n")

# Analysis Functions
def save_loss_plot(train_losses, val_losses, positional_embedding):
    train_losses_clean = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in train_losses]
    val_losses_clean = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in val_losses]

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_clean, label='Training Loss')
    plt.plot(val_losses_clean, label='Validation Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    path = os.path.join(OUTPUT_DIR, f'loss_curve_{positional_embedding}_{DATE_TIME}.png') 
    plt.savefig(path)
    plt.close()

def visualize_test_samples(loader, model, epoch, positional_embedding):
    model.eval()

    try:
        data, valid_mask = next(iter(loader))
    except StopIteration:
        return

    data = data.to(DEVICE)
    valid_mask = valid_mask.to(DEVICE)
    mask = generate_multiscale_mask(data.size(0), size=PATCH_SIZE, target_ratio=MASK_RATIO)
    masked_data = data.clone()
    masked_data[:, 0, :, :] = masked_data[:, 0, :, :] * mask[:, 0, :, :]
    # 18 data + cloud mask + valid mask
    model_input = torch.cat([masked_data, mask, valid_mask], dim=1)

    with torch.no_grad():
        pred_img = model(model_input)

    data = data.cpu().numpy()
    pred_img = pred_img.cpu().numpy()

    # Add unmasked pixels back to the prediction for better visualization 
    mask_np = mask.cpu().numpy()
    blended_pred = np.where(mask_np[:, 0] == 1.0, data[:, 0], pred_img[:, 0])

    # Grab 3 example prediction batches 
    n = 3
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))

    # Create custom color bar map to grey out masked areas 
    custom_mask_cmap = plt.get_cmap('viridis').copy()
    custom_mask_cmap.set_bad(color='gray')

    for i in range(n):
        # Get the vmin and vmax from the real data to set a fixed colorbar to compare model outputs and real data to
        real_data_vmin = data[i, 0].min() 
        real_data_vmax = data[i, 0].max()
        #real_data_vmin = -3.0 
        #real_data_vmax = 3.0

        # Recreate masked data map with NaN values so we can color them in for the plot
        masked_plot = np.where(mask_np[i, 0] == 1.0, data[i, 0], np.nan)

        axes[i][0].imshow(masked_plot, cmap=custom_mask_cmap, vmin=real_data_vmin, vmax=real_data_vmax)
        axes[i][0].set_title("Masked Input (Snowdepth)")

        axes[i][1].imshow(blended_pred[i], cmap='viridis', vmin=real_data_vmin, vmax=real_data_vmax)
        axes[i][1].set_title("Prediction (Snowdepth)")

        axes[i][2].imshow(data[i, 0], cmap='viridis', vmin=real_data_vmin, vmax=real_data_vmax)
        axes[i][2].set_title("Ground Truth (Snowdepth)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'prediction_samples_epoch_{epoch}_{positional_embedding}_{DATE_TIME}.png')
    plt.savefig(path)
    plt.close()

def calculate_metrics_in_meters(loader, model, stats):
    model.eval()
    mse_accum = 0.0
    mae_accum = 0.0
    count = 0
    res_mean = stats['mean'][0]
    res_std = stats['std'][0]

    with torch.no_grad():
        for batch_data, batch_valid_mask in loader:
            batch_data = batch_data.to(DEVICE)
            batch_valid_mask = batch_valid_mask.to(DEVICE)

            mask = generate_multiscale_mask(batch_data.size(0), size=PATCH_SIZE, target_ratio=MASK_RATIO)
            masked_data = batch_data.clone()
            masked_data[:, 0, :, :] = masked_data[:, 0, :, :] * mask[:, 0, :, :]
            # 18 data + cloud mask + valid mask
            model_input = torch.cat([masked_data, mask, batch_valid_mask], dim=1)

            pred_img = model(model_input)
            pred_res = pred_img[:, 0, :, :]
            true_res = batch_data[:, 0, :, :]

            # Denormalize
            pred_meters = (pred_res * res_std) + res_mean
            pred_meters = torch.clamp(pred_meters, min=0.0)
            true_meters = (true_res * res_std) + res_mean

            # Only evaluate pixels that were cloud-masked and are inside basin
            cloud_mask_bool = (mask[:, 0, :, :] == 0)
            boundary_valid_bool = (batch_valid_mask.squeeze(1) == 1.0)
            final_eval_mask = cloud_mask_bool & boundary_valid_bool

            if final_eval_mask.sum() > 0:
                diff = pred_meters[final_eval_mask] - true_meters[final_eval_mask]
                mse_accum += (diff ** 2).sum().item()
                mae_accum += diff.abs().sum().item()
                count += final_eval_mask.sum().item()

    if count == 0: return 0, 0
    rmse = np.sqrt(mse_accum / count)
    mae = mae_accum / count
    return rmse, mae

CHANNEL_NAMES = [
    'Snow Depth (ASO)', 'Day of Year', 'SNODAS_1', 'SNODAS_2', 'SNODAS_3', 'SNODAS_4',
    'DTM', 'Curvature (General)', 'Curvature (Plan)', 'TPI_9', 'TPI_101',
    'Slope', 'Eastness', 'Northness', 'R', 'G', 'B', 'Canopy Height'
]

def channel_ablation_study(loader, model, stats):
    """Zero out each input channel one at a time and measure RMSE change."""
    print("\n" + "=" * 70)
    print("  CHANNEL ABLATION STUDY")
    print("  Zeroing each channel and measuring RMSE impact")
    print("=" * 70)

    baseline_rmse, baseline_mae = calculate_metrics_in_meters(loader, model, stats)
    print(f"\n  Baseline RMSE: {baseline_rmse:.4f} m | MAE: {baseline_mae:.4f} m\n")

    results = []
    for ch in range(18):
        model.eval()
        mse_accum = 0.0
        mae_accum = 0.0
        count = 0
        res_mean = stats['mean'][0]
        res_std = stats['std'][0]

        with torch.no_grad():
            for batch_data, batch_valid_mask in loader:
                batch_data = batch_data.to(DEVICE)
                batch_valid_mask = batch_valid_mask.to(DEVICE)

                mask = generate_multiscale_mask(batch_data.size(0), size=PATCH_SIZE, target_ratio=MASK_RATIO)
                masked_data = batch_data.clone()
                masked_data[:, 0, :, :] = masked_data[:, 0, :, :] * mask[:, 0, :, :]

                # Zero out the ablated channel
                masked_data[:, ch, :, :] = 0.0

                model_input = torch.cat([masked_data, mask, batch_valid_mask], dim=1)
                pred_img = model(model_input)
                pred_res = pred_img[:, 0, :, :]
                true_res = batch_data[:, 0, :, :]

                pred_meters = (pred_res * res_std) + res_mean
                pred_meters = torch.clamp(pred_meters, min=0.0)
                true_meters = (true_res * res_std) + res_mean

                cloud_mask_bool = (mask[:, 0, :, :] == 0)
                boundary_valid_bool = (batch_valid_mask.squeeze(1) == 1.0)
                final_eval_mask = cloud_mask_bool & boundary_valid_bool

                if final_eval_mask.sum() > 0:
                    diff = pred_meters[final_eval_mask] - true_meters[final_eval_mask]
                    mse_accum += (diff ** 2).sum().item()
                    mae_accum += diff.abs().sum().item()
                    count += final_eval_mask.sum().item()

        if count == 0:
            rmse, mae = 0.0, 0.0
        else:
            rmse = np.sqrt(mse_accum / count)
            mae = mae_accum / count

        delta_rmse = rmse - baseline_rmse
        name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Channel {ch}"
        results.append((ch, name, rmse, delta_rmse))
        print(f"  Ch {ch:2d} ({name:20s}): RMSE = {rmse:.4f} m  |  Delta = {delta_rmse:+.4f} m")

    # Sort by impact (largest RMSE increase = most important)
    results.sort(key=lambda x: x[3], reverse=True)
    print(f"\n  Ranked by importance (RMSE increase when removed):")
    print(f"  {'Rank':<5} {'Channel':<22} {'Delta RMSE':>12}")
    print(f"  {'-'*5} {'-'*22} {'-'*12}")
    for rank, (ch, name, rmse, delta) in enumerate(results, 1):
        marker = " ***" if delta > 0.01 else ""
        print(f"  {rank:<5} {name:<22} {delta:>+12.4f} m{marker}")

    # Save results to file
    ablation_path = os.path.join(OUTPUT_DIR, f'channel_ablation_{positional_embedding}_{DATE_TIME}.txt')
    with open(ablation_path, 'w') as f:
        f.write(f"Baseline RMSE: {baseline_rmse:.4f} m | MAE: {baseline_mae:.4f} m\n\n")
        f.write(f"{'Rank':<5} {'Ch':>3} {'Channel':<22} {'RMSE':>10} {'Delta RMSE':>12}\n")
        f.write(f"{'-'*55}\n")
        for rank, (ch, name, rmse, delta) in enumerate(results, 1):
            f.write(f"{rank:<5} {ch:>3} {name:<22} {rmse:>10.4f} {delta:>+12.4f}\n")
    print(f"\n  Results saved to {ablation_path}")
    print("=" * 70)

    return results

def force_metadata_match(source_path, reference_path, output_path):
    # Code to force the source raster to adopt the CRS and bounding box of the reference raster'''
    try:
        with rasterio.open(reference_path) as ref_ds:
            target_crs = ref_ds.crs
            target_bounds = ref_ds.bounds

        with rasterio.open(source_path) as src_ds:
            data = src_ds.read()
            out_profile = src_ds.profile.copy()
                
            new_transform = from_bounds(
                target_bounds.left, 
                target_bounds.bottom, 
                target_bounds.right, 
                target_bounds.top, 
                src_ds.width, 
                src_ds.height
            )

            NODATA_VALUE = -9999
            out_profile.update({
                'crs': target_crs,
                'transform': new_transform,
                'driver': 'GTiff',
                'nodata': NODATA_VALUE
            })

            with rasterio.open(output_path, 'w', **out_profile) as dst_ds:
                dst_ds.write(data)
                    
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit()

def make_output_plots(pos_emb_name, date_time_str, ground_truth_date, residual_ground_truth_file_path, residual_prediction_file_path, input_mask=None):
    print(f"Comparing:\n  Pred: {residual_prediction_file_path}\n  True: {residual_ground_truth_file_path}")

    pred_img = tiff.imread(residual_prediction_file_path).flatten()
    true_img = tiff.imread(residual_ground_truth_file_path).flatten()

    # Do not check masked out lakes and padding
    valid_mask = (true_img != -9999.0) & (~np.isnan(true_img))

    # In partial mode, only score pixels the model predicted (input_mask == 0),
    # not the unmasked pixels where ground truth was visible to the model
    if input_mask is not None:
        input_mask_flat = input_mask.flatten()
        valid_mask = valid_mask & (input_mask_flat <= 0.5)

    y_true = true_img[valid_mask]
    y_pred = pred_img[valid_mask]

    # Calculate residuals
    residuals = y_pred - y_true
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    print(f"Image size: {np.sum(valid_mask)} pixels:")
    print(f" \tRMSE: {rmse:.4f}")
    print(f" \tMAE:  {mae:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter Plot (True vs Predicted)
    hb = axes[0].hexbin(y_true, y_pred, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'w--', linewidth=2, label="Perfect Prediction")
    axes[0].set_title(f"Prediction Accuracy\n(Log Density)")
    axes[0].set_xlabel("Ground Truth Snow Depth(m)")
    axes[0].set_ylabel("Predicted Snow Depth (m)")
    axes[0].legend()
    cb = fig.colorbar(hb, ax=axes[0])
    cb.set_label('Log10(Pixel Count)')

    # Snow Depth Prediction Plot (True vs Error)
    axes[1].scatter(y_true[::100], residuals[::100], alpha=0.3, s=1, c='blue')
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set_title("Predicted vs. Ground Truth Snow Depth Plot\n(Downsampled 100x)")
    axes[1].set_xlabel("Ground Truth Snow Depth (m)")
    axes[1].set_ylabel("Error (Pred - True) (m)")

    # Error Histogram (bias frequency)
    axes[2].hist(residuals, bins=100, color='gray', log=True)
    axes[2].axvline(0, color='black', linestyle='--')
    axes[2].set_title(f"Error Distribution\nMean Bias: {np.mean(residuals):.4f}m")
    axes[2].set_xlabel("Error (m)")
    axes[2].set_ylabel("Frequency (Log Scale)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"prediction_analysis_for_{ground_truth_date}_{pos_emb_name}_{date_time_str}.png"))
    print(f"Saved analysis plot to {os.path.join(OUTPUT_DIR, f'prediction_analysis_for_{ground_truth_date}_{pos_emb_name}_{date_time_str}.png')}")
    #plt.show()

    # Spatial Error Map
    h, w = tiff.imread(residual_ground_truth_file_path).shape
    spatial_error = np.zeros((h, w))
    spatial_error[:] = np.nan

    full_true = tiff.imread(residual_ground_truth_file_path)
    full_pred = tiff.imread(residual_prediction_file_path)

    mask_2d = (full_true != -9999.0)
    spatial_error[mask_2d] = full_pred[mask_2d] - full_true[mask_2d]

    fig_map, ax_map = plt.subplots(figsize=(10, 8))
    im = ax_map.imshow(spatial_error, cmap='bwr', vmin=-4.0, vmax=4.0)
    if input_mask is not None:
        input_visible = (input_mask > 0.1) & ~np.isnan(input_mask)
        overlay = np.zeros((*spatial_error.shape, 4), dtype=np.float32)
        overlay[input_visible] = [1.0, 0.85, 0.0, 0.75]
        ax_map.imshow(overlay)
    fig_map.colorbar(im, ax=ax_map, label="Error (m) [Red = Overpredict, Blue = Underpredict]")
    ax_map.set_title("Spatial Error Map" + (" [Gold = Snow Depth Input to Model]" if input_mask is not None else ""))
    fig_map.savefig(os.path.join(OUTPUT_DIR, f"spatial_error_map_for_{ground_truth_date}_{pos_emb_name}_{date_time_str}.png"))
    plt.close(fig_map)
    print(f"Saved spatial map to {os.path.join(OUTPUT_DIR, f'spatial_error_map_for_{ground_truth_date}_{pos_emb_name}_{date_time_str}.png')}")

    return rmse, mae, int(np.sum(valid_mask))

def create_gaussian_window(size, sigma=None):
    # 2D Gaussian window for smooth tile blending
    if sigma is None:
        sigma = size / 4.0
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g1d = torch.exp(-coords**2 / (2 * sigma**2))
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    return g2d.numpy()

def predict_basin(folder_path, ground_truth_date, model, global_stats, blind=True):

    print(f"Processing validation folder: {folder_path}")
    model.eval()
    try:
        res_files = glob.glob(os.path.join(folder_path, "*ASO*") )
        doy_files = glob.glob(os.path.join(folder_path, "*acqdoy*") )
        snodas_files = sorted(glob.glob(os.path.join(folder_path, "*SNODAS*") ))

        if not res_files or not doy_files or len(snodas_files) != 4: return

        res_img = tiff.imread(res_files[0]).astype(np.float32)
        doy_img = tiff.imread(doy_files[0]).astype(np.float32)
        snodas_imgs = [tiff.imread(f).astype(np.float32) for f in snodas_files]

        res_img[res_img == -9999] = 0.0 
        res_img[np.isnan(res_img)] = 0.0
        doy_img[doy_img == -9999] = 0.0
        doy_img[np.isnan(doy_img)] = 0.0

        for i in range(len(snodas_imgs)):
            snodas_imgs[i][snodas_imgs[i] == -9999] = 0.0
            snodas_imgs[i][np.isnan(snodas_imgs[i])] = 0.0

        # Get static topo input data
        static_imgs = []
        static_filenames = [
            FN_DTM, FN_CURV_GEN, FN_CURV_PLAN, FN_TPI_9, FN_TPI_101,
            FN_SLOPE, FN_EAST, FN_NORTH,
            FN_R, FN_G, FN_B, FN_CANOPY
        ]

        for fn in static_filenames:
            path = os.path.join(STATIC_DIR, fn)
            s_img = tiff.imread(path).astype(np.float32)
            s_img[s_img == -9999] = 0.0
            s_img[np.isnan(s_img)] = 0.0

            static_imgs.append(s_img)

        # Stack data and normalize
        all_channels = [res_img, doy_img] + snodas_imgs + static_imgs
        full_stack = np.stack(all_channels, axis=0).astype(np.float32)
        channels, h, w = full_stack.shape

        prediction_accum = np.zeros((h, w), dtype=np.float32)
        count_accum = np.zeros((h, w), dtype=np.float32)

        mean = global_stats['mean'][:, None, None]
        std = global_stats['std'][:, None, None]
        norm_stack = (full_stack - mean) / std
        norm_stack = np.nan_to_num(norm_stack, nan=0.0)

        # Build valid mask from original data (before 0-fill)
        raw_orig = tiff.imread(res_files[0]).astype(np.float32)
        valid_mask_full = np.ones_like(raw_orig)
        valid_mask_full[raw_orig == -9999.0] = 0.0
        valid_mask_full[np.isnan(raw_orig)] = 0.0

        gaussian_window = create_gaussian_window(PATCH_SIZE)

        mask_vote_accum = np.zeros((h, w), dtype=np.float32)
        tile_hit_accum = np.zeros((h, w), dtype=np.float32)

        overlap = 64
        stride = PATCH_SIZE - overlap
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + PATCH_SIZE, h)
                x_end = min(x + PATCH_SIZE, w)
                y_start = y_end - PATCH_SIZE
                x_start = x_end - PATCH_SIZE

                if y_start < 0:
                    y_start = 0
                    y_end = PATCH_SIZE

                if x_start < 0:
                    x_start = 0
                    x_end = PATCH_SIZE

                tile = norm_stack[:, y_start:y_end, x_start:x_end].copy()
                tile_valid_mask = valid_mask_full[y_start:y_end, x_start:x_end]
                tile_valid_mask = tile_valid_mask[None, :, :]

                if blind:
                    tile_mask = np.zeros((1, tile.shape[1], tile.shape[2]), dtype=np.float32)
                    tile[0] = 0.0
                else:
                    mask_torch = generate_multiscale_mask(1, size=PATCH_SIZE, target_ratio=MASK_RATIO)
                    tile_mask = mask_torch.squeeze(0).cpu().numpy()
                    tile[0] = tile[0] * tile_valid_mask[0] * tile_mask[0]
                    mask_vote_accum[y_start:y_end, x_start:x_end] += tile_mask[0]
                    tile_hit_accum[y_start:y_end, x_start:x_end] += 1.0

                tile_input = np.concatenate([tile, tile_mask, tile_valid_mask], axis=0)
                tensor_tile = torch.from_numpy(tile_input).unsqueeze(0).float().to(DEVICE)

                with torch.no_grad():
                    pred_img = model(tensor_tile)

                pred_np = pred_img.squeeze().cpu().numpy()
                prediction_accum[y_start:y_end, x_start:x_end] += pred_np * gaussian_window
                count_accum[y_start:y_end, x_start:x_end] += gaussian_window

        final_prediction = prediction_accum / np.maximum(count_accum, 1e-6)
        res_mean = global_stats['mean'][0]
        res_std = global_stats['std'][0]
        final_prediction = (final_prediction * res_std) + res_mean
        final_prediction = np.clip(final_prediction, 0.0, None)

        if blind:
            real_prediction = np.where(valid_mask_full == 1.0, final_prediction, -9999.0)
            return_mask = None
        else:
            avg_visible = mask_vote_accum / np.maximum(tile_hit_accum, 1e-6)
            blended = np.where(avg_visible > 0.5, raw_orig, final_prediction)
            real_prediction = np.where(valid_mask_full == 1.0, blended, -9999.0)
            return_mask = np.where(valid_mask_full == 1.0, avg_visible, np.nan)

        run_label = "blind" if blind else "partial"
        output_path = os.path.join(OUTPUT_DIR, f"model_prediction_for_date_{ground_truth_date}_{positional_embedding}_{DATE_TIME}_{run_label}.tif")
        tiff.imwrite(output_path, real_prediction)
        print(f"Analysis-Ready Prediction saved to {output_path}")
        return [output_path, res_files[0], return_mask]

    except Exception as e:
        print(f"Error predicting basin: {e}")

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # CHECK GPU IS BEING USED!
    print(f"Using device: {DEVICE}")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    if not os.path.exists(DATA_DIR):
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    all_day_folders = sorted([f.path for f in os.scandir(DATA_DIR) if f.is_dir()])
    verify_data_integrity(all_day_folders)

    folders_2024 = sorted([f for f in all_day_folders if os.path.basename(f)[:4] == '2024'])
    folders_2025 = sorted([f for f in all_day_folders if os.path.basename(f)[:4] == '2025'])

    val_folders  = folders_2024[-3:]
    test_folders = folders_2025[-2:]

    holdout = set(val_folders + test_folders)
    train_folders = [f for f in all_day_folders if f not in holdout]

    print(f"Train: {len(train_folders)} folders | Val: {val_folders} | Test: {test_folders}")

    global_stats = compute_dataset_stats(train_folders)

    static_imgs = load_static_images()
    print(f"Loaded {len(static_imgs)} static terrain images")

    train_dataset = GeoFolderDataset(train_folders, stats=global_stats, static_imgs=static_imgs, patch_size=PATCH_SIZE, augment=True,  samples_per_file=SAMPLES_PER_FILE)
    val_dataset   = GeoFolderDataset(val_folders,   stats=global_stats, static_imgs=static_imgs, patch_size=PATCH_SIZE, augment=False, samples_per_file=150)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=5, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=1, prefetch_factor=1, pin_memory=True, persistent_workers=True)

    for positional_embedding in POSITIONAL_EMBEDDING_LIST:
        print(f'\n\nRunning model with {positional_embedding} positional embedding!\n')

        # create output folder by current datetime to be unique
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, positional_embedding, DATE_TIME)

        if os.path.exists(OUTPUT_DIR):
            raise RuntimeError('ERROR: folder {} already exists, please wait a minute to run this code again.')
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        # TensorBoard writer
        tb_log_dir = os.path.join(OUTPUT_DIR, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_log_dir)

        # ViT Model
        model = MaskedAutoencoderViT(
            pos_emb_selected=positional_embedding,
            img_size=PATCH_SIZE,
            patch_size=VIT_PATCH_SIZE,
            in_chans=IN_CHANS,
            out_chans=OUT_CHANS,
            embed_dim=EMBED_DIM,
            depth=NUM_LAYERS,
            num_heads=NUM_HEADS,
            decoder_embed_dim=DECODER_EMBED_DIM,
            decoder_depth=NUM_DECODER_LAYERS,
            decoder_num_heads=NUM_HEADS,
            dropout=DROPOUT
        ).to(DEVICE)
        model = torch.compile(model)

        # Verify model shapes with a dummy forward pass
        dummy_input = torch.randn(2, IN_CHANS, PATCH_SIZE, PATCH_SIZE, device=DEVICE)
        dummy_output = model(dummy_input)
        print(f"Shape check: input {list(dummy_input.shape)}: output {list(dummy_output.shape)}")
        assert dummy_output.shape == (2, OUT_CHANS, PATCH_SIZE, PATCH_SIZE), \
            f"Shape mismatch! Expected [2, {OUT_CHANS}, {PATCH_SIZE}, {PATCH_SIZE}], got {list(dummy_output.shape)}"
        del dummy_input, dummy_output
        torch.cuda.empty_cache()

        optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

        # Initialize Scaler
        scaler = GradScaler('cuda')

        # Use warmup and decay scheduler to prevent gradient explosion
        scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
        scheduler_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[WARMUP_EPOCHS])

        train_losses = []
        val_losses = []

        # Print TensorBoard command before training starts
        print("")
        print("=" * 80)
        print("  TRAINING IS STARTING")
        print(f"  Train: {len(train_dataset)} samples: {len(train_loader)} batches/epoch")
        print(f"  Val:   {len(val_dataset)} samples: {len(val_loader)} batches/epoch")
        print("")
        print("  To monitor training in real time, open a new terminal and run:")
        print(f"  tensorboard --logdir=\"{tb_log_dir}\"")
        print("")
        print("  Then open http://localhost:6006 in your browser")
        print("=" * 80)
        print("", flush=True)

        # Write each epoch loss score to a text file
        with open(os.path.join(OUTPUT_DIR, f'epoch_loss_output_{positional_embedding}_{DATE_TIME}.txt'), 'w') as epoch_output_file:
            write_hyperparameters_to_file(epoch_output_file, positional_embedding)
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(EPOCHS):
                epoch_start = time()
                model.train()
                total_train_loss = 0
                total_train_grad_loss = 0

                num_batches = len(train_loader)
                # Process batches
                for batch_idx, (batch_data, batch_valid_mask) in enumerate(train_loader):
                    batch_data = batch_data.to(DEVICE)
                    batch_valid_mask = batch_valid_mask.to(DEVICE)

                    # Target snow depth channel 0 [B, 1, H, W]
                    target = batch_data[:, 0:1, :, :]

                    # Generate fake cloud masks
                    mask = generate_multiscale_mask(batch_data.size(0), size=PATCH_SIZE, target_ratio=MASK_RATIO)
                    masked_data = batch_data.clone()
                    masked_data[:, 0, :, :] = masked_data[:, 0, :, :] * mask[:, 0, :, :]
                    model_input = torch.cat([masked_data, mask, batch_valid_mask], dim=1)
                    cloud_eval_mask = batch_valid_mask * (1.0 - mask)

                    optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(model_input)

                        res_mean_t = torch.tensor(global_stats['mean'][0], device=DEVICE)
                        res_std_t = torch.tensor(global_stats['std'][0], device=DEVICE)
                        pred_meters = (pred * res_std_t) + res_mean_t

                        mse_loss = masked_mse_loss(pred, target, cloud_eval_mask)
                        grad_loss = gradient_loss(pred, target, cloud_eval_mask)
                        relu_penalty = (cloud_eval_mask * F.relu(-pred_meters) ** 2).sum() / (cloud_eval_mask.sum() + 1e-6)
                        loss = mse_loss + (GRADIENT_LOSS_WEIGHT * grad_loss) + (RELU_PENALTY_WEIGHT * relu_penalty)

                    if torch.isnan(loss):
                        print(f"nan loss encountered at batch {batch_idx}")
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_train_loss += loss.detach()
                    total_train_grad_loss += grad_loss.detach()

                    if batch_idx % 25 == 0:
                        elapsed = time() - epoch_start
                        print(f"  Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | {elapsed:.0f}s", flush=True)

                scheduler.step()
                avg_train_loss = total_train_loss / max(1, len(train_loader))
                avg_train_grad_loss = total_train_grad_loss / max(1, len(train_loader))
                train_losses.append(avg_train_loss)

                # Run over validation data
                model.eval()
                total_val_loss = 0
                total_val_grad_loss = 0
                with torch.no_grad():
                    for batch_data, batch_valid_mask in val_loader:
                        batch_data = batch_data.to(DEVICE)
                        batch_valid_mask = batch_valid_mask.to(DEVICE)

                        target = batch_data[:, 0:1, :, :]

                        mask = generate_multiscale_mask(batch_data.size(0), size=PATCH_SIZE, target_ratio=MASK_RATIO)
                        masked_data = batch_data.clone()
                        masked_data[:, 0, :, :] = masked_data[:, 0, :, :] * mask[:, 0, :, :]
                        model_input = torch.cat([masked_data, mask, batch_valid_mask], dim=1)
                        cloud_eval_mask = batch_valid_mask * (1.0 - mask)

                        pred = model(model_input)
                        res_mean_t = torch.tensor(global_stats['mean'][0], device=DEVICE)
                        res_std_t = torch.tensor(global_stats['std'][0], device=DEVICE)
                        pred_meters = (pred * res_std_t) + res_mean_t
                        mse_loss = masked_mse_loss(pred, target, cloud_eval_mask)
                        grad_loss = gradient_loss(pred, target, cloud_eval_mask)
                        relu_penalty = (cloud_eval_mask * F.relu(-pred_meters) ** 2).sum() / (cloud_eval_mask.sum() + 1e-6)
                        val_loss = mse_loss + GRADIENT_LOSS_WEIGHT * grad_loss + RELU_PENALTY_WEIGHT * relu_penalty

                        if not torch.isnan(val_loss):
                            total_val_loss += val_loss.item()
                            total_val_grad_loss += grad_loss.item()

                avg_val_loss = total_val_loss / max(1, len(val_loader))
                avg_val_grad_loss = total_val_grad_loss / max(1, len(val_loader))
                val_losses.append(avg_val_loss)

                current_lr = scheduler.get_last_lr()[0]

                # TensorBoard logging
                writer.add_scalar('Loss/train_total', avg_train_loss, epoch)
                writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
                writer.add_scalar('Loss/train_gradient', avg_train_grad_loss, epoch)
                writer.add_scalar('Loss/val_gradient', avg_val_grad_loss, epoch)
                writer.add_scalar('Hyperparameters/learning_rate', current_lr, epoch)

                epoch_end = time()
                epoch_output = f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | Runtime: {epoch_end - epoch_start:.1f}s\n"
                epoch_output_file.write(epoch_output)
                print(epoch_output, end='')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    model_save_path = os.path.join(OUTPUT_DIR, f'best_snow_model_{positional_embedding}_epoch_{epoch + 1}_{DATE_TIME}.pth')
                    torch.save(model.state_dict(), model_save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                        break

                # Output sample plots every 10 epochs
                if (epoch + 1) % 10 == 0:
                    visualize_test_samples(val_loader, model, epoch+1, positional_embedding)
                    # Log sample prediction images to TensorBoard
                    sample_img_path = os.path.join(OUTPUT_DIR, f'prediction_samples_epoch_{epoch+1}_{positional_embedding}_{DATE_TIME}.png')
                    if os.path.exists(sample_img_path):
                        from PIL import Image
                        img = np.array(Image.open(sample_img_path))
                        writer.add_image('Predictions/samples', img, epoch, dataformats='HWC')

            save_loss_plot(train_losses, val_losses, positional_embedding)

            print("\nTraining Finished. Loading best model for final evaluation...")
            model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
            model.eval()

            print("Calculating RMSE and MAE...")
            rmse, mae = calculate_metrics_in_meters(val_loader, model, global_stats)

            final_val_results = f"\nFinal Validation Results: RMSE: {rmse:.4f} m, MAE: {mae:.4f} m"
            epoch_output_file.write(final_val_results)
            print(final_val_results)

            # Log final metrics to TensorBoard
            writer.add_scalar('Metrics/RMSE_meters', rmse, EPOCHS)
            writer.add_scalar('Metrics/MAE_meters', mae, EPOCHS)

            # Channel ablation test
            print("\nRunning channel ablation test...")
            channel_ablation_study(val_loader, model, global_stats)

            # Run on all validation folders
            if len(test_folders) > 0:
                for test_folder in test_folders:
                    ground_truth_date = os.path.basename(test_folder)
                    for blind in [True, False]:
                        path_list = predict_basin(test_folder, ground_truth_date, model, global_stats, blind=blind)
                        if path_list is None:
                            print(f"WARNING: predict_basin returned None for {ground_truth_date}, skipping")
                            continue
                        residual_prediction_file_path = path_list[0]
                        residual_ground_truth_file_path = path_list[1]
                        retrieved_input_mask = path_list[2]
                        run_label = "blind" if blind else "partial"
                        residual_prediction_file_fixed_crs_path = os.path.join(OUTPUT_DIR, f"model_prediction_for_test_date_{ground_truth_date}_{positional_embedding}_{DATE_TIME}_{run_label}_fixed.tif")
                        force_metadata_match(residual_prediction_file_path, residual_ground_truth_file_path, residual_prediction_file_fixed_crs_path)
                        make_output_plots(positional_embedding, DATE_TIME, ground_truth_date, residual_ground_truth_file_path, residual_prediction_file_fixed_crs_path, input_mask=retrieved_input_mask)
                        os.remove(residual_prediction_file_path)
            else:
                raise RuntimeError(f'AHHH! No validation files were assigned! AHHH! How did you even get this far?')

            print(f"SUCCESS: Model weights saved to {model_save_path}")

            # SAVE THE STATS
            stats_save_path = os.path.join(OUTPUT_DIR, f'dataset_stats_{positional_embedding}_{DATE_TIME}.npy')
            np.save(stats_save_path, global_stats)
            print(f"SUCCESS: Dataset statistics saved to {stats_save_path}")

        writer.close()