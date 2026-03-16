"""
Homework 4 - Learning Visual Features with CNNs
CSCI1430 - Computer Vision
Brown University

Hyperparameters for all tasks. Adjust these if you want to experiment.
"""
MAX_PARAMS = 10_000_000   # max number of parameters in your model
                          # we will count them directly from your .pt in the autograder

# Task 0: End-to-end scene classification
ENDTOEND_IMAGE_SIZE = 224
ENDTOEND_EPOCHS = 25
ENDTOEND_LR = 1e-4
ENDTOEND_BATCH_SIZE = 32

# Task 1: Rotation pretraining (1 image)
ROTATION_EPOCHS = 50
ROTATION_LR = 5e-2
ROTATION_BATCH_SIZE = 32
ROTATION_CROP_SIZE = 224
ROTATION_NUM_CROPS = 50_000

# Task 2: Transfer evaluation
TRANSFER_IMAGE_SIZE = 224
TRANSFER_EPOCHS = 15
TRANSFER_HEAD_LR = 1e-3   # learning rate for the linear head
TRANSFER_ENCODER_LR = 1e-4 # learning rate for the encoder (finetune only)
TRANSFER_BATCH_SIZE = 32

# Extra Credit: Classification pretraining (2 images)
CLASSIFY_EPOCHS = 50
CLASSIFY_LR = 5e-2
CLASSIFY_BATCH_SIZE = 32
CLASSIFY_CROP_SIZE = 224
CLASSIFY_NUM_CROPS = 50_000

# Extra Credit: Define any other hyperparameters you need
