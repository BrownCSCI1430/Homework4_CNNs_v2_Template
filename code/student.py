"""
Homework 4 - CNNs: Learning Visual Features
CSCI1430 - Computer Vision
Brown University

Task 0: Design a CNN and train it end-to-end on 15-scene classification.
Task 1: Learn features via self-supervised pretraining, without labels.
Task 2: Transfer pretrained features to 15-scenes — can you beat Task 0?

    uv run python main.py --task <task_name>
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

import hyperparameters as hp
from helpers import visualize_filters, save_filter_frame, make_filter_video


BROWN_ID = 000000000
torch.manual_seed(BROWN_ID)
np.random.seed(BROWN_ID)


# ========================================================================
#  SceneDataset — loads the 15-scenes dataset (given, do not modify)
# ========================================================================

class SceneDataset:
    """Wraps torchvision.ImageFolder for train/val/test splits."""

    def __init__(self, data_dir, batch_size=hp.BATCH_SIZE, image_size=hp.IMAGE_SIZE):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        self.classes = train_set.classes
        self.num_classes = len(self.classes)


# ========================================================================
#  TASK 0: End-to-end scene classification
#
#  Design a CNN and train it from scratch on 15-scene classification.
#  This is your baseline — later you'll try to beat it with pretraining.
# ========================================================================

# --- Part A: Training loop (used by all tasks) ---

def train_loop(model, train_loader, optimizer, criterion, epochs,
               device, val_loader=None, label="", on_epoch_end=None):
    """Train a model and optionally evaluate on a validation set each epoch.

    TODO: Implement the training loop. For each epoch:
        a. Set model to training mode.
        b. Loop over batches: move to device, forward pass, compute loss,
           backward pass, optimizer step. Track running accuracy and loss.
        c. If val_loader is provided, evaluate: set model to eval mode,
           compute val accuracy with torch.no_grad(), append to val_accs.
        d. Print a status line each epoch (format shown below).
        e. If on_epoch_end is not None, call it: on_epoch_end(epoch, model)

    Print format (so output is consistent):
        f"[{label}] Epoch {epoch+1}/{epochs}  Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
        (append f"  Val: {val_acc:.3f}" if val_loader is provided)

    Args:
        model: nn.Module to train
        train_loader: DataLoader for training data
        optimizer: torch.optim optimizer
        criterion: loss function (e.g., nn.CrossEntropyLoss())
        epochs: number of training epochs
        device: torch.device ('cpu' or 'cuda')
        val_loader: optional DataLoader for validation
        label: string prefix for print output
        on_epoch_end: optional callback, called as on_epoch_end(epoch, model)

    Returns:
        List of validation accuracies (float, one per epoch).
        Empty list if val_loader is None.
    """
    val_accs = []
    # TODO: implement training loop
    return val_accs


# --- Part B: Design your SceneClassifier ---

class SceneClassifier(nn.Module):
    """Your CNN architecture for 15-scene classification.

    TODO: Design a CNN with these requirements:
        - self.encoder: nn.Module — the convolutional feature extractor
        - self.head: nn.Module — the classification head
            (typically: AdaptiveAvgPool2d(1) -> Flatten -> Linear -> ... -> Linear(num_classes))
        - self.encoder_channels: int — number of output channels from encoder
        - Total parameters must be <= hp.MAX_PARAMS (10,000,000)
        - forward(x) returns logits of shape (batch_size, num_classes)

    Hint: See the handout for architecture design guidance. Start simple.
    More parameters does NOT mean better performance on small datasets!
    """

    def __init__(self, num_classes=15):
        super().__init__()
        # TODO: define self.encoder_channels, self.encoder, self.head
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


# --- Part C: Train your SceneClassifier end-to-end ---

def t0_endtoend(classify_15scenes_data, device, approaches):
    """Train SceneClassifier from scratch on 15-scenes.

    TODO:
        1. Create a SceneClassifier and move it to device.
        2. Verify param count: sum(p.numel() for p in classifier.parameters()) <= hp.MAX_PARAMS
        3. Create an optimizer (e.g., Adam with lr=hp.ENDTOEND_LR).
        4. Call train_loop with hp.ENDTOEND_EPOCHS epochs,
           passing classify_15scenes_data.val_loader for validation.
        5. Save classifier.state_dict() to approaches['endtoend'].weights
        6. Save the val accuracy list (np.save) to approaches['endtoend'].curve
    """
    pass


# ========================================================================
#  TASK 1: Self-supervised pretraining
#
#  Learn visual features from just one or two images — no class labels!
#  The key idea: generate thousands of random crops, then train a CNN
#  to predict which rotation was applied to each crop.
# ========================================================================

# --- Part A: CropRotationDataset ---

class CropRotationDataset(Dataset):
    """Dataset that generates random crops from images for rotation prediction.

    TODO: Implement __init__, __len__, and __getitem__.

    __init__ should:
        - Load all .jpg images from data_dir as PIL images.
          Handle both flat directories and subdirectory layouts (like ImageFolder).
        - Store images in self.pil_images, class indices in self.class_indices.
        - Set up a random crop transform (e.g., RandomResizedCrop(crop_size, ...)).
        - Create self.train_loader = DataLoader(self, batch_size=batch_size, ...)
        - Set self.num_classes = 4 if rotation mode, else number of image classes.

    __len__: return self.num_crops (the number of crops per epoch, NOT image count).

    __getitem__: for index idx:
        1. Pick a random image from self.pil_images.
        2. Extract a random crop from the image.
        3. Convert to tensor.
        4. If rotation mode: pick a random rotation (0, 90, 180, or 270 degrees),
           apply it, return (tensor, rotation_id) where rotation_id is 0-3.
        5. If classification mode: return (tensor, class_index).

    Optional augmentations: color jitter, horizontal flip, aggressive crop
    scales (see Asano et al. 2020). Even simple fixed-size random crops
    work well for learning filters.
    """

    def __init__(self, data_dir, num_crops=hp.NUM_CROPS, crop_size=hp.CROP_SIZE,
                 rotation=True, batch_size=hp.BATCH_SIZE):
        # TODO
        raise NotImplementedError

    def __len__(self):
        # TODO
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO
        raise NotImplementedError


# --- Part B: Design your pretraining encoder ---
#
# Design a small encoder for self-supervised pretraining.
# Create your own nn.Module class here. Requirements:
#
#   - self.layers must be an nn.Sequential
#   - self.layers[0] must be a Conv2d(3, ...) — needed for filter visualization
#   - End with AdaptiveAvgPool2d(1) so output shape is (batch, channels, 1, 1)
#
# See the handout for design guidance. Keep it MUCH simpler than SceneClassifier.
# Our solution uses ~1M parameters. The challenge is learning good features
# with a minimal architecture — not building a big network.
#
# Write your encoder class below:


# --- Part C: Filter visualization callback (given, do not modify) ---

def _conv1_diagnostics(encoder, w0, w_prev, epoch, frame_dir):
    """Print conv1 weight diagnostics and save filter frames (raw + delta)."""
    w = encoder.layers[0].weight.data.cpu()
    diff_from_init = (w - w0).abs().mean().item()
    diff_from_prev = (w - w_prev[0]).abs().mean().item()
    mag = w.abs().mean().item()
    w_std = w.std().item()

    grad_str = ""
    if encoder.layers[0].weight.grad is not None:
        grad_norm = encoder.layers[0].weight.grad.data.abs().mean().item()
        grad_str = f"  grad={grad_norm:.6f}"

    print(f"  conv1: diff_init={diff_from_init:.4f}  diff_ep={diff_from_prev:.5f}  "
          f"|w|={mag:.4f}  std={w_std:.4f}  ratio={diff_from_init/mag:.0%}{grad_str}",
          flush=True)

    save_filter_frame(encoder, epoch, output_dir=frame_dir)

    # Save delta filter frame (w - w0) — shows what the network learned
    delta_dir = frame_dir + '_delta'
    os.makedirs(delta_dir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    delta = w - w0
    n = delta.shape[0]
    cols, rows = 8, (n + 7) // 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            f = delta[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f.permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.suptitle(f'Learned Delta (w - w0) -- Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(delta_dir, f'epoch_{epoch:03d}.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    w_prev[0] = w.clone()


# --- Part D: Rotation pretraining ---

def t1_rotation(rotation_data, device, approaches):
    """Train your encoder with rotation prediction on a single image.

    TODO:
        1. Create your encoder and build a rotation prediction model:
               model = nn.Sequential(encoder, nn.Flatten(1), nn.Linear(out_dim, 4))
           where out_dim is the number of channels your encoder outputs.
        2. Create an optimizer (SGD with lr=hp.ROTATION_LR, momentum=0.9 works well).
        3. Set up the filter visualization callback:
               w0 = encoder.layers[0].weight.data.cpu().clone()
               w_prev = [w0.clone()]
               def my_callback(epoch, model):
                   _conv1_diagnostics(encoder, w0, w_prev, epoch, 'filter_frames_rotation')
                   visualize_filters(encoder, save_path='conv1_filters_rotation.png')
        4. Call train_loop with hp.ROTATION_EPOCHS epochs.
        5. Make filter videos:
               make_filter_video('filter_frames_rotation', 'filters_rotation.mp4')
               make_filter_video('filter_frames_rotation_delta', 'filters_rotation_delta.mp4')
        6. Save encoder.state_dict() to approaches['rotation'].weights
    """
    pass


# --- Part E: Classification pretraining ---

def t1_classify(classify_data, device, approaches):
    """Train your encoder with binary classification on two images.

    TODO: Same pattern as t1_rotation, but:
        - The linear head has classify_data.num_classes outputs (2, not 4).
        - Use hp.CLASSIFY_LR and hp.CLASSIFY_EPOCHS.
        - Save frames to 'filter_frames_classify' (and '_delta' variant).
        - Save encoder to approaches['classify'].weights
        - Save filters to 'conv1_filters_classify.png'
    """
    pass


# ========================================================================
#  TASK 2: Transfer evaluation
#
#  Take your pretrained encoder and test it on 15-scene classification.
#  Can pretrained features match or beat your end-to-end SceneClassifier?
# ========================================================================

def t2_transfer(classify_15scenes_data, device, approaches):
    """Evaluate pretrained encoder features on 15-scene classification.

    Run three experiments (hp.TRANSFER_EPOCHS epochs each) and save val curves:

    TODO:
        1. Frozen pretrained probe:
           - Create your encoder, load weights from approaches['rotation'].weights
             (or approaches['classify'].weights — your choice).
           - FREEZE the encoder: for p in encoder.parameters(): p.requires_grad = False
           - Put encoder in eval mode: encoder.eval()
           - Build model: nn.Sequential(encoder, nn.Flatten(1), nn.Linear(out_dim, num_classes))
           - Optimize ONLY the linear head: Adam(model[-1].parameters(), lr=hp.TRANSFER_HEAD_LR)
           - Train and save val accuracies to approaches['frozen_pretrained'].curve
           - Save model.state_dict() to approaches['frozen_pretrained'].weights

        2. Frozen random probe (control):
           - Same as above, but with a fresh (untrained) encoder.
           - This proves your pretrained features are better than random.
           - Save to approaches['frozen_random'].curve

        3. Finetune pretrained:
           - Create encoder, load pretrained weights (do NOT freeze).
           - Use separate learning rates for encoder vs head:
               optimizer = Adam([
                   {'params': encoder.parameters(), 'lr': hp.TRANSFER_ENCODER_LR},
                   {'params': head.parameters(), 'lr': hp.TRANSFER_HEAD_LR},
               ])
           - Save to approaches['finetune'].curve
    """
    pass
