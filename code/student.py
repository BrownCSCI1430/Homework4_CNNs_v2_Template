"""
Homework 4 - CNNs
CSCI1430 - Computer Vision
Brown University

Task 0: Load the data
  1. SceneDataset                       - 15 scenes train/test/val with classification labels
  2. CropRotationDataset                - Random crops for pretraining, including rotation

Tasks 1 & 2:
  1. train_loop                         - Write a generic training loop for all tasks
  2. t1_rotation_1img                   - Train an ExampleEncoder on the self-supervised rotation task, 
                                          and visualize the first-layer features
  3. t2_classify_2img                   - Train an ExampleEncoder on the supervised classification task
                                          and visualize the first-layer features

Tasks 3:
  1. t3_pretrained_classify_15scenes    - Train simple classification heads for the 15 scene tasks
                                          from your pretrained features.                             
                              
Tasks 4:
  1. SceneClassifier                    - Design your own encoder and classifier design for 15 scenes
  2. t4_endtoend_classify_15scenes      - Train your classifier and evalute it.

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

# ========================================================================
#  Set your Brown Banner ID here. This seeds all random number generators
#  so your trained weights are unique to you.
# ========================================================================
BROWN_ID = 000000000  # ← replace with your Banner ID
torch.manual_seed(BROWN_ID)
np.random.seed(BROWN_ID)

# ========================================================================
#  TASK 0: Data loading
#
#  Two patterns for loading data in PyTorch:
#
#  1. SceneDataset — uses ImageFolder (a built-in Dataset) + DataLoader.
#     The standard approach when your data is already in folders, and any
#     data processes and augmentations are applied with Transforms.
#
#  2. CropRotationDataset — a custom Dataset subclass with __getitem__.
#     This loads a single image and generates crops at different rotations.
#     This seems like a data augmentation, but isn't: the label in this 
#     self-supervised task is whichever rotation was applied. 
#     But Transforms can't produce labels.
# ========================================================================

class SceneDataset:
    """Load the 15-scenes dataset using ImageFolder — the standard PyTorch way.

    This class organizes the three splits (train/val/test) 
    and their DataLoaders in one place.

    Expects data_dir to contain train/, val/, and test/ subdirectories,
    each with one subfolder per class (ImageFolder format).

    After construction, provides:
        .train_loader  -- DataLoader for training set (shuffled)
        .val_loader    -- DataLoader for validation set
        .test_loader   -- DataLoader for test set
        .classes       -- list of class name strings
        .num_classes   -- number of classes

    Arguments:
        data_dir   -- path to data directory (must contain train/, val/, test/)
        batch_size -- batch size for DataLoaders (default 32)
        image_size -- resize images to this square size (default 224)
    """

    def __init__(self, data_dir, batch_size=hp.BATCH_SIZE, image_size=hp.IMAGE_SIZE):
        # TODO:
        # 1. Set self.classes and self.num_classes.
        #
        # 2. Create train, val, and test datasets using datasets.ImageFolder
        #    on os.path.join(data_dir, 'train'), 'val', and 'test'.
        #
        # 3. Each ImageFolder needs a transform: Resize((image_size, image_size)) → ToTensor()
        #    Here, you can include any other data augmentations as needed.
        #
        # 3. Wrap each dataset in a DataLoader (shuffle=True for train,
        #    False for val/test) →
        #    self.train_loader, self.val_loader, self.test_loader

        raise NotImplementedError("TODO: implement SceneDataset.__init__")


class CropRotationDataset(Dataset):
    """
    Create a dataset of random crops from images, and optionally rotate them.
    Note: Not about farming 👩‍🌾🌾🥕

    Loads all images from data_dir, which may contain class subfolders.
    The label depends on the mode:

        rotation=True  → Randomly rotates the crop.
                         The label is which rotation was applied (0-3).
        rotation=False → Label is the class index (subfolder index).
                         I.E., for single-images dataset, class Coast = 0, Street = 1

    Arguments:
        data_dir   -- path to a directory of images (with or without class subfolders)
        batch_size -- batch size for the DataLoader (default 32)
        crop_size  -- spatial size of each crop (default 224)
        num_crops  -- total number of crops to generate per epoch
        rotation   -- if True, apply random rotation and return rotation label;
                      if False, return the class label (default True)

    After construction, provides:
        .train_loader  -- DataLoader for this dataset (shuffled)
        .classes       -- list of class name strings
        .num_classes   -- number of classes

    Note: There is no .test_loader or .val_loader - the data are too small.
    """

    def __init__(self, data_dir, batch_size=hp.BATCH_SIZE, num_crops=hp.NUM_CROPS, crop_size=hp.CROP_SIZE, rotation=True):
        # TODO:
        # 1. Set self.num_crops, self.crop_size, self.rotation.
        #
        # 2. Set self.classes and self.num_classes.
        #    rotation=True  → num_classes = 4 (one per rotation)
        #    rotation=False → num_classes = number of class subfolders
        #
        # 3. Load the source images into self.images as numpy arrays.
        #    These are what __getitem__ will crop from.
        #    Note: Most datasets are too large to load all at once.
        #    We have a tiny dataset - just one or two images. So, it's ok to load everything.
        #
        # 4. Wrap this Dataset in a DataLoader for batching/shuffling:
        #    self.train_loader = DataLoader(self, batch_size=batch_size,
        #                                  shuffle=True, num_workers=0)
        #
        # 5. In preparation for __getitem__, create a ToTensor() transform 
        #    This converts an HxWx3 uint8 numpy array into a 3xHxW float tensor in [0,1]. Used by __getitem__.

        raise NotImplementedError("TODO: implement CropRotationDataset.__init__")

    def __len__(self):
        return self.num_crops

    def __getitem__(self, idx):
        """Return a random crop from a random source image.

        Returns:
            crop  -- (3, crop_size, crop_size) tensor
            label -- if rotation=True:  integer in {0, 1, 2, 3} (rotation class)
                     if rotation=False: integer class index (which folder)
        """

        # TODO:
        # 1. Pick a random image from self.images.
        # 2. Extract a random crop, rotate at random as needed.
        # 2b.Add any other augmentations that might help.
        # 3. Make it a Tensor using your ToTensor().
        # 4. Define the label.

        raise NotImplementedError("TODO: implement CropRotationDataset.__getitem__")
    

# ========================================================================
#  TASKS 1, 2, & 3
# ========================================================================

# ========================================================================
#  ExampleEncoder
#
#  An AlexNet-inspired encoder with a large 11×11 first convolution 
#  so the learned filters are big enough to visualize as images.
#
#  Input:  (B, 3, 224, 224)
#  Output: (B, 256, 1, 1)
#
#  This encoder works out of the box for Tasks 1, 2, and 3.
#  Study it as an example before designing your own SceneClassifier
#  for Task 4.
# ========================================================================
class ExampleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 224×224 → 56×56
            nn.Conv2d(in_channels=3, out_channels=32, 
                      kernel_size=11, stride=4, padding=5), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            
            # 56×56 → 28×28
            nn.Conv2d(in_channels=32, out_channels=64, 
                      kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 28×28 → 14×14
            nn.Conv2d(in_channels=64, out_channels=128, 
                      kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 14×14 → 7×7
            nn.Conv2d(in_channels=128, out_channels=256, 
                      kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 7×7 → 3×3
            nn.Conv2d(in_channels=256, out_channels=256, 
                      kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 3×3 → 1×1
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.layers(x)


# ========================================================================
#  Training loop — used by all tasks
# ========================================================================
def train_loop(model, train_loader, optimizer, criterion, epochs,
               device, val_loader=None, label="", on_epoch_end=None):
    """Train a model for classification. Used by all tasks.

    Each epoch:
      1. Train: forward pass → loss → backward → optimizer step
      2. Print training accuracy
      3. If val_loader: evaluate and print validation accuracy
      4. If on_epoch_end: call on_epoch_end(epoch, model)

    Args:
        on_epoch_end: optional callback called at the end of each epoch.
                      Signature: on_epoch_end(epoch, model).
                      Used by Tasks 1 & 2 to save filter visualizations.

    Returns a list of validation accuracies (one per epoch).
    If val_loader is None, returns an empty list.
    """

    # TODO:
    # 1. For each epoch:
    #    a. Set model to training mode.
    #    b. For each (images, labels) batch in train_loader:
    #       - Move images and labels to device
    #       - Forward pass: logits = model(images)
    #       - Compute loss = criterion(logits, labels)
    #       - Zero gradients, backpropagate, optimizer step
    #       - Track correct predictions and total count
    #    c. Print training accuracy for this epoch.
    #
    # 2. If val_loader is not None:
    #    a. Set model to eval mode.
    #    b. With torch.no_grad(), loop over val_loader and compute accuracy.
    #    c. Append val accuracy to a list, print it.
    #
    # 3. If on_epoch_end is not None, call it: on_epoch_end(epoch, model)
    #
    # 4. Return the list of validation accuracies.

    raise NotImplementedError("TODO: implement train_loop")


# ========================================================================
#  Task 1: Rotation prediction — 1 image, self-supervised
#
#  Train the ExampleEncoder with a single-layer classification head
#  to predict which of 4 rotations was applied to crops from a SINGLE image.
#
#  After training, visualize the first-layer filters. You should see
#  oriented edge detectors (Gabor-like filters) emerge from just one
#  image — the same low-level features that appear in ImageNet-trained
#  networks.
# ========================================================================

def t1_rotation_1img(rotation_1img_data, device, approaches):

    # TODO:
    # 1. Create an ExampleEncoder and attach a head.
    #    Say, a single linear layer with 256 neurons and num_classes output (4).
    #
    # 2. Move it to device.
    #
    # 3. Create an optimizer and set a learning rate -> hyperparameters.py has one.
    #
    # 4. Define a callback that saves filter frames each epoch:
    #        def my_callback(epoch, model):
    #            save_filter_frame(encoder, epoch, output_dir='filter_frames_rotation')
    #
    # 5. Call train_loop with on_epoch_end=my_callback.
    #
    # 6. After training, make the video:
    #        make_filter_video('filter_frames_rotation', 'filters_rotation.gif')
    #
    # 7. Save encoder weights: torch.save(encoder.state_dict(), approaches['rotation'].weights)
    #
    # 8. Visualize final filters: visualize_filters(encoder, save_path='conv1_filters_rotation.png')

    raise NotImplementedError("TODO: implement t1_rotation_1img")


# ========================================================================
#  Task 2: Binary classification — 2 images, supervised
#
#  Train the ExampleEncoder to classify crops as "Street" or "Coast".
#  This uses real labels, but only two images — isolating the effect of
#  supervision vs. data quantity.
#
#  Compare the learned filters to Task 1. Both tasks produce edge
#  detectors, but the supervision signal differs (rotation vs. class).
# ========================================================================

def t2_classify_2img(classify_2img_data, device, approaches):

    # TODO:
    # 1. Create an ExampleEncoder and attach a classification head.
    #    Say, a single linear layer with 256 neurons.
    #
    # 2. Move it to device.
    #
    # 3. Create an optimizer and set a learning rate -> hyperparameters.py has one.
    #
    # 4. Define a callback that saves filter frames each epoch:
    #        def my_callback(epoch, model):
    #            save_filter_frame(encoder, epoch, output_dir='filter_frames_classify')
    #
    # 5. Call train_loop with on_epoch_end=my_callback.
    #
    # 6. After training, make the video:
    #        make_filter_video('filter_frames_classify', 'filters_classify.gif')
    #
    # 7. Save encoder weights: torch.save(encoder.state_dict(), approaches['classify'].weights)
    #
    # 8. Visualize final filters: visualize_filters(encoder, save_path='conv1_filters_classify_2img.png')

    raise NotImplementedError("TODO: implement t2_classify_2img")


# ========================================================================
#  Task 3: Pretrained classification — use features from Tasks 1 & 2
#
#  Load the encoders you trained in Tasks 1 & 2, attach a fresh
#  classification head, and fine-tune on 15-scenes.
#
#  How well do features learned from 1-2 images transfer to a real
#  classification task?
# ========================================================================

def t3_pretrained_classify_15scenes(classify_15scenes_data, device, approaches):

    # TODO:
    # Use your pretrained weights without their heads from tasks 1 and 2 to 
    # train new heads for the 15-scenes classification dataset.
    # And, train a new head on _purely random weights_ as a baseline.
    # 
    # The three approaches:
    #   1. Random weights baseline — fresh ExampleEncoder.
    #      Save curve -> approaches['random'].curve
    #   2. Rotation-pretrained — load from approaches['rotation'].weights
    #      Save curve -> approaches['rotation'].curve
    #   3. Classify-pretrained — load from approaches['classify'].weights
    #      Save curve -> approaches['classify'].curve
    #
    # To load pretrained weights:
    #   encoder.load_state_dict(torch.load(path, weights_only=True))

    raise NotImplementedError("TODO: implement t3_pretrained_classify_15scenes")


# ========================================================================
#  Task 4
#
#  A free design task for creating an encoder for the 15-scene dataset.
#  How good can you make it?
#
#  Total parameters must be ≤ 10M.
# ========================================================================

class SceneClassifier(nn.Module):
    """Your classifier for 15-scenes.

    Input:  (B, 3, H, W)   — typically 224x224
    Output: (B, num_classes)

    Total parameters (this entire model) must be ≤ 10M.

    Architecture requirements:
      - self.encoder: nn.Module that maps (B, 3, H, W) → (B, C, h, w)
        
      - self.head: nn.Module that maps (B, C, h, w) → (B, num_classes)
        Could start with AdaptiveAvgPool2d(1) to handle varying spatial dims.

      - self.encoder_channels: int — number of output channels C from encoder.

    Study ExampleEncoder above for an example of an encoder.
    """

    def __init__(self, num_classes=15):
        super().__init__()

        # TODO: Design your encoder and classification head.
        #
        # Requirements:
        #   - self.encoder: processes images of any spatial size → feature maps
        #   - self.head: AdaptiveAvgPool2d(1) → Flatten → Linear layers → num_classes
        #   - self.encoder_channels: int, the number of channels your encoder outputs
        #   - Total params ≤ 10M
        #
        # Hint: look at how ExampleEncoder builds conv blocks, then design
        # your own architecture. You can use the same pattern or try something
        # different — it's your design.

        raise NotImplementedError("TODO: implement SceneClassifier.__init__")

    def forward(self, x):
        # TODO: Pass x through self.encoder, then self.head.

        raise NotImplementedError("TODO: implement SceneClassifier.forward")


# ========================================================================
#  Task 4: End-to-end scene classification — your architecture
#
#  IMPORTANT: Save your filter visualizations and accuracy numbers
#  from Tasks 1 & 2 before running this — you'll need them for
#  your reslog.md

#  Design your own SceneClassifier and train it on 15-scenes.
#  Target: >= 65% validation accuracy, <= 10M parameters, <= 25 epochs.
#
# ========================================================================

def t4_endtoend_classify_15scenes(classify_15scenes_data, device, approaches):

    # TODO:
    # 1. Create your SceneClassifier(num_classes=...) and move to device.
    #
    # 2. Count and print total parameters. Assert <= hp.MAX_PARAMS.
    #
    # 3. Create an optimizer with lr=hp.ENDTOEND_CLASSIFY_LR.
    #
    # 4. Call train_loop for hp.ENDTOEND_CLASSIFY_EPOCHS epochs with val_loader.
    #
    # 5. Save classifier weights and learning curve:
    #    torch.save(classifier.state_dict(), approaches['endtoend'].weights)
    #    np.save(approaches['endtoend'].curve, accs)

    raise NotImplementedError("TODO: implement t4_endtoend_classify_15scenes")
