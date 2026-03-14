"""
Homework 4 - CNNs: Pretraining an Encoder
CSCI1430 - Computer Vision
Brown University

Hyperparameters for all tasks. Adjust these if you want to experiment.
"""
MAX_PARAMS = 10_000_000   # max number of parameters in your model 
                          # we will count them directly from your .pt in the autograder

# -- Data --
BATCH_SIZE = 32           # batch size for all tasks
IMAGE_SIZE = 224          # all images (crops and scenes) are this size
CROP_SIZE = 224           # size of crops
NUM_CROPS = 5000          # number of random crops per epoch

# -- Task 1: Rotation (1 image, self-supervised) --
ROTATION_1IMG_EPOCHS = 50
ROTATION_1IMG_LR = 1e-3

# -- Task 2: Classify (2 images, supervised) --
CLASSIFY_2IMG_EPOCHS = 50
CLASSIFY_2IMG_LR = 1e-3

# -- Task 3: Pretrained classification (fine-tune head on 15-scenes) --
PRETRAINED_CLASSIFY_EPOCHS = 5
PRETRAINED_CLASSIFY_LR = 1e-4

# -- Task 4: End-to-end classification (your architecture) --
ENDTOEND_CLASSIFY_EPOCHS = 25
ENDTOEND_CLASSIFY_LR = 1e-4
