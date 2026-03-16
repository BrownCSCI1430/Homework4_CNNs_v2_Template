"""
Homework 4 - Learning Visual Features with CNNs
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
from helpers import visualize_filters, make_filter_video, make_filter_callback

BANNER_ID = 000000000 # <- replace with your Banner ID; drop the 'B' prefix and any leading 0s.
torch.manual_seed(BANNER_ID)
np.random.seed(BANNER_ID)


# ========================================================================
#  SceneDataset — loads the 15-scenes dataset (given, do not modify)
#
class SceneDataset:
    """Load the 15-scenes dataset using ImageFolder (given, do not modify).

    Organizes train/val/test splits and their DataLoaders in one place.
    Expects data_dir to contain train/, val/, and test/ subdirectories,
    each with one subfolder per class (ImageFolder format).

    Hyperparameters are defined in hp.ENDTOEND_*

    Arguments:
        data_dir   -- path to dataset (must contain train/, val/, test/)
        batch_size -- batch size for DataLoaders
        image_size -- resize images to this square size

    After construction, provides:
        .train_loader  -- DataLoader for training set (shuffled)
        .val_loader    -- DataLoader for validation set
        .test_loader   -- DataLoader for test set
        .classes       -- list of class name strings
        .num_classes   -- number of classes
    """

    def __init__(self, data_dir, batch_size=hp.ENDTOEND_BATCH_SIZE, image_size=hp.ENDTOEND_IMAGE_SIZE):

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
        test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = 0 if os.name == 'nt' else 4)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers = 0 if os.name == 'nt' else 4)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers = 0 if os.name == 'nt' else 4)
        self.classes = train_set.classes
        self.num_classes = len(self.classes)


# ========================================================================
#  TASK 0: End-to-end scene classification
#
#  Design a CNN and train it from scratch on 15-scene classification.
#  This is your baseline — later you'll try to beat it with pretraining.
# ========================================================================

# Part A: Training loop (used by all tasks)
#
def train_loop(model, train_loader, optimizer, loss, epochs,
               device, val_loader=None, tasklabel="", on_epoch_end=None):
    """Train a model and optionally evaluate on a validation set each epoch.

    Arguments:
        model:          nn.Module to train
        train_loader:   DataLoader for training data
        optimizer:      torch.optim optimizer
        loss:           loss function (e.g., nn.CrossEntropyLoss())
        epochs:         number of training epochs
        device:         torch.device passed from main.py
        val_loader:     optional DataLoader for validation
        tasklabel:      string prefix for print output
        on_epoch_end:   optional callback, called as on_epoch_end(epoch, model)

    Returns:
        List of training accuracies     (float, one per epoch).
        List of validation accuracies   (float, one per epoch); empty if val_loader is None.
    """
    train_accs = []
    val_accs = []

    # TODO: Implement the training loop. For each epoch:
    #     a. Set model to training mode.

    #     b. Loop over batches: move to device, forward pass, compute loss,
    #        backward pass, optimizer step. Track running accuracy and loss.

    #     c. If val_loader is provided, evaluate: set model to eval mode,
    #        compute val accuracy with torch.no_grad(), append to val_accs.

    #     d. Print a status line each epoch (format shown below).
    #         f"[{tasklabel}] Epoch {epoch+1}/{epochs}  Train: {train_acc:.3f}  Loss: {avg_loss:.4f}"
    #         (append f"  Val: {val_acc:.3f}" if val_loader is provided)

    #     e. If on_epoch_end is not None, call it at the end of an epoch: 
    #         on_epoch_end(epoch, model)

    return train_accs, val_accs


# Part B: Design your SceneClassifier
#
class SceneClassifier(nn.Module):
    """Your CNN architecture for 15-scene classification.

    Hint: See the handout for architecture design guidance. Start simple.
    More parameters does NOT mean better performance on small datasets!

    Your final model must be under 10 M parameters.
    """

    def __init__(self, num_classes=15):
        super().__init__()

        # TODO: Design a CNN with these requirements:
        #     - self.encoder: nn.Module — the convolutional feature extractor
        #                     This should end with AdaptiveAvgPool2d(1)
        #                     so it works at any input resolution
        #     - self.head: nn.Module — a single-layer classification head
        #         (Flatten -> Linear(encoder_channels, num_classes))
        #     - self.encoder_channels: int — number of output channels from encoder
        #     - forward(x) returns logits of shape (batch_size, num_classes)

        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


# Part C: Train your SceneClassifier end-to-end
#
def t0_endtoend(classify_15scenes_data, device, approaches):
    """Train SceneClassifier from scratch on 15-scenes.

    Hyperparameters are defined in hp.ENDTOEND_*
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)

    # TODO:
    #     1. Create a SceneClassifier and move it to device.
    #     2. Create an optimizer and a loss.
    #     3. Call train_loop with hp.ENDTOEND_EPOCHS epochs,
    #        passing classify_15scenes_data.val_loader for validation.
    #     4. Save classifier.state_dict() to approaches['endtoend'].weights
    #     5. Save the val accuracy list to approaches['endtoend'].curve_val
    #     6. Save the train accuracy list to approaches['endtoend'].curve_train

    pass


# ========================================================================
#  TASK 1: Self-supervised pretraining
#
#  Learn visual features from just one or two images — no class labels!
#  The key idea: generate thousands of random crops, then train a CNN
#  to predict which rotation was applied to each crop.
# ========================================================================

# Part A: CropRotationDataset
#
class CropRotationDataset(Dataset):
    """Create a dataset of random rotated crops from images.
    Note: Not about farming. 👩‍🌾🌾🌽

    Hyperparameters are defined in hp.ROTATION_*

    Important: For speed, implement all operations using pytorch functions
               after moving the image to the device (GPU).

    Arguments:
        data_dir   -- path to a directory of images (with or without class subfolders)
        num_crops  -- total number of crops to generate per epoch
        crop_size  -- spatial size of each crop
        rotation   -- if True (default), apply random rotation and return rotation label
        batch_size -- batch size for the DataLoader

    After construction, provides:
        .train_loader  -- DataLoader for this dataset (shuffled)
        .classes       -- list of class name strings
        .num_classes   -- number of classes

    Note: Unlike SceneDataset, there is no .test_loader or .val_loader — the data are too small.

    Simple fixed-size random crops work well for learning filters.
    Optional augmentations: color jitter, horizontal flip, crops at different
    scales (see Asano et al. 2020).

    [EXTRA CREDIT] To implement a classification pretraining task:

        - Hyperparameters are defined in hp.CLASSIFY_*
        - Input data live in two directories - Street and Coast
        - rotation argument -- if False, return the class label not the rotation label
        - All data augmentations might still apply...

    """

    def __init__(self, data_dir, num_crops=hp.ROTATION_NUM_CROPS,
                 crop_size=hp.ROTATION_CROP_SIZE, rotation=True,
                 batch_size=hp.ROTATION_BATCH_SIZE):
        # TODO:
        # 1. Set self.num_crops, self.crop_size, self.rotation, self.batch_size
        #
        # 2. Set self.classes and self.num_classes.
        #    rotation=True  -> num_classes = 4 (one per rotation)
        #    rotation=False -> num_classes = number of class subfolders
        #
        # 3. Load source images and transfer them to device as tensors.
        #    Note: Most datasets are too large to load all at once.
        #    We have a tiny dataset — just one or two images. So, it's ok.
        #
        # 4. Wrap this Dataset in a DataLoader for batching/shuffling:
        #    self.train_loader = DataLoader(self, batch_size=batch_size,
        #                                  shuffle=True, num_workers=0)

        raise NotImplementedError("TODO: implement CropRotationDataset.__init__")

    def __len__(self):
        return self.num_crops

    def __getitem__(self, idx):
        """Return a random crop from a random source image.

        Returns:
            crop  -- (3, crop_size, crop_size) float32 tensor in [0, 1]
            label -- if rotation=True:  integer in {0, 1, 2, 3} (rotation class)
                     [Extra Credit] if rotation=False: integer class index {0, 1} (which directory, Street or Coast)
        """
        # TODO:
        # 1. Pick a random source image (as a tensor, already on device).
        # 2. Extract a random crop and rotate it at random as needed.
        # 3. Add any other augmentations that might help.
        # 4. Define the label.

        raise NotImplementedError("TODO: implement CropRotationDataset.__getitem__")


# Part B: Design your pretraining encoder
#
class PretrainingEncoder(nn.Module):
    """Design an encoder for self-supervised pretraining.

    Our solution uses ~1M parameters. The challenge is learning good features
    with a minimal architecture — not building a big network.
    """
    def __init__(self):
        super().__init__()

        # TODO:
        # Create your own nn.Module class here. Requirements:
        #    - self.layers must be an nn.Sequential
        #    - self.layers[0] must be a Conv2d(3, ...) — needed for filter visualization
        #      We use 11 x 11 kernels for the first layer to make this easily visible.
        #    - End with AdaptiveAvgPool2d(1) so output shape is (batch, channels, 1, 1)

        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

# Part C: Rotation pretraining
#
def t1_rotation(rotation_data, device, approaches):
    """Train your encoder with rotation prediction on a single image.
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)

    # TODO:
    #     1. Create your encoder and build a rotation prediction model:
    #            model = nn.Sequential(encoder, nn.Flatten(1), nn.Linear(out_dim, 4))
    #        where out_dim is the number of channels your encoder outputs.

    #     2. Create an optimizer and a loss.

    #     3. Create the filter visualization callback:
    #            callback = make_filter_callback(encoder, 'results/filter_frames_rotation',
    #                                            'results/conv1_filters_rotation.png')

    #     4. Call train_loop with hp.ROTATION_EPOCHS epochs and on_epoch_end=callback.

    #     5. Make filter videos:
    #            make_filter_video('results/filter_frames_rotation', 'results/filters_rotation.mp4')
    #            make_filter_video('results/filter_frames_rotation_delta', 'results/filters_rotation_delta.mp4')

    #     6. Save the training accuracy list to approaches['rotation'].curve_train

    #     7. Save encoder.state_dict() to approaches['rotation'].weights
    #
    pass

# ========================================================================
#  TASK 2: Transfer evaluation
#
#  Take your pretrained encoder and test it on 15-scene classification.
#  Can pretrained features match or beat your end-to-end SceneClassifier?
# ========================================================================

def t2_transfer(classify_15scenes_data, device, approaches):
    """Evaluate pretrained encoder features on 15-scene classification.

    Run three experiments (hp.TRANSFER_EPOCHS epochs each) and save your val curves and model weights:

    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)

    # TODO:

    #     1. Frozen random features + linear classification head
    #        - Create an untrained encoder with random initial weights.
    #        - FREEZE the encoder: for p in encoder.parameters(): p.requires_grad = False
    #        - Put encoder in eval mode: encoder.eval()
    #        - Build a simple linear classification head: nn.Sequential(encoder, nn.Flatten(1), nn.Linear(out_dim, num_classes))
    #        - Optimize ONLY the linear head: your_optimizer(model[-1].parameters(), lr=hp.TRANSFER_HEAD_LR)
    #        - Save train/val accuracies to approaches['frozen_random'].curve_train / .curve_val
    #        - Save model.state_dict() to approaches['frozen_random'].weights

    #     2. Frozen pretrained features + linear classification head:
    #        - Same workflow as above, but...
    #        - Create your encoder with loaded weights from approaches['rotation'].weights
    #        - FREEZE the encoder, put it in eval mode, build the classification head, optimize it.
    #        - Save train/val accuracies to approaches['frozen_pretrained'].curve_train / .curve_val
    #        - Save model.state_dict() to approaches['frozen_pretrained'].weights

    #     3. Finetune your pretrained features + linear classification head:
    #        - Same as above, but this time do NOT freeze your pretrained weights.
    #        - You can use separate learning rates for encoder vs head, e.g.,:
    #            optimizer = your_choice([
    #                {'params': encoder.parameters(), 'lr': hp.TRANSFER_ENCODER_LR},
    #                {'params': head.parameters(), 'lr': hp.TRANSFER_HEAD_LR},
    #            ])
    #        - Save train/val accuracies to approaches['finetune'].curve_train / .curve_val
    #        - Save model.state_dict() to approaches['finetune'].weights

    pass


# Extra Credit: Classification pretraining
#
def t1_classify(classify_data, device, approaches):
    """Train your encoder with binary classification on two images.
    Add a new path to t2_transfer and use approaches['classify'] for saving outputs.
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)

    pass


# Extra Credit: Open-ended self-supervised pretraining
#
def t1_ec_pretrain(device, approaches):
    """Train your encoder with any self-supervised approach you design.
    Add a new path to t2_transfer and use approaches['ec_frozen'] for saving outputs.

    Feel free to define a new architecture.
    The goal: maximize frozen features + linear head
    accuracy on 15-scenes (evaluated via the leaderboard on a secret test set).
    """
    # Reproducible initialization — do not remove
    torch.manual_seed(BANNER_ID)

    pass
