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
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple

import torch

from student import (
    SceneDataset, CropRotationDataset,
    t0_endtoend, t1_rotation, t1_classify, t2_transfer,
)
from hyperparameters import *


# ========================================================================
#  File naming conventions
# ========================================================================

Approach = namedtuple('Approach', ['label', 'weights', 'curve'])

APPROACHES = {
    'endtoend':          Approach('End-to-end (from scratch)',    'endtoend_classifier.pt',    'curve_endtoend.npy'),
    'rotation':          Approach('Rotation-pretrained encoder',  'rotation_encoder.pt',       'curve_rotation_encoder.npy'),
    'classify':          Approach('Classify-pretrained encoder',  'classify_encoder.pt',       'curve_classify_encoder.npy'),
    'frozen_random':     Approach('Frozen random probe',          'frozen_random.pt',          'curve_frozen_random.npy'),
    'frozen_pretrained': Approach('Frozen pretrained probe',      'frozen_pretrained.pt',      'curve_frozen_pretrained.npy'),
    'finetune':          Approach('Finetune pretrained',          'finetune.pt',               'curve_finetune.npy'),
}


# ========================================================================
#  Dispatch
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4: Learning Visual Features with CNNs")
    parser.add_argument('--task', required=True,
                        choices=['t0_endtoend',
                                 't1_rotation', 
                                 't1_classify', # Extra credit
                                 't2_transfer'])
    parser.add_argument('--data', default=os.path.join('..', 'data'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.chdir(sys.path[0])

    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ---- Task 0: End-to-end scene classification ----
    if args.task == 't0_endtoend':
        classify_15scenes_data = SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=ENDTOEND_IMAGE_SIZE, batch_size=ENDTOEND_BATCH_SIZE,
        )
        t0_endtoend(classify_15scenes_data, device, APPROACHES)

    # ---- Task 1: Rotation pretraining (1 image) ----
    elif args.task == 't1_rotation':
        rotation_data = CropRotationDataset(
            os.path.join(args.data, 'single-images', 'train', 'Street'),
            num_crops=ROTATION_NUM_CROPS, crop_size=ROTATION_CROP_SIZE,
            rotation=True, batch_size=ROTATION_BATCH_SIZE,
        )
        t1_rotation(rotation_data, device, APPROACHES)

    # ---- Task 1b: Extra Credit: Classification pretraining (2 images) ----
    elif args.task == 't1_classify':
        classify_data = CropRotationDataset(
            os.path.join(args.data, 'single-images', 'train'),
            num_crops=CLASSIFY_NUM_CROPS, crop_size=CLASSIFY_CROP_SIZE,
            rotation=False, batch_size=CLASSIFY_BATCH_SIZE,
        )
        t1_classify(classify_data, device, APPROACHES)

    # ---- Task 2: Transfer evaluation ----
    elif args.task == 't2_transfer':
        classify_15scenes_data = SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=ENDTOEND_IMAGE_SIZE, batch_size=TRANSFER_BATCH_SIZE,
        )
        t2_transfer(classify_15scenes_data, device, APPROACHES)
