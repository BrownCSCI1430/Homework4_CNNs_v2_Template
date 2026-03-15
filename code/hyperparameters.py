"""
Homework 4 - CNNs: Learning Visual Features
CSCI1430 - Computer Vision
Brown University

Hyperparameters for all tasks. Adjust these if you want to experiment.
"""
MAX_PARAMS = 10_000_000   # max number of parameters in your model
                          # we will count them directly from your .pt in the autograder

# -- Data --
BATCH_SIZE = 32           # batch size for all tasks
IMAGE_SIZE = 224          # all images (crops and scenes) are this size
CROP_SIZE = 224           # size of crops for pretraining
NUM_CROPS = 50_000        # number of random crops per epoch

# -- Task 0: End-to-end scene classification --
ENDTOEND_EPOCHS = 25
ENDTOEND_LR = 1e-4

# -- Task 1: Self-supervised pretraining --
ROTATION_EPOCHS = 50      # rotation prediction (1 image)
ROTATION_LR = 0.05        # SGD with momentum works well here
CLASSIFY_EPOCHS = 50      # binary classification (2 images)
CLASSIFY_LR = 0.05

# -- Task 2: Transfer evaluation --
TRANSFER_EPOCHS = 15
TRANSFER_HEAD_LR = 1e-3   # learning rate for the linear head
TRANSFER_ENCODER_LR = 1e-4 # learning rate for the encoder (finetune only)
