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
    t0_endtoend, t1_rotation, t1_classify, t1_ec_pretrain, t2_transfer,
)
from hyperparameters import *


# ========================================================================
#  File naming conventions
# ========================================================================

Approach = namedtuple('Approach', ['label', 'weights', 'curve_train', 'curve_val'])

APPROACHES = {
    'endtoend':          Approach('End-to-end (from scratch)',   'results/endtoend_classifier.pt',  'results/train_endtoend.npy',          'results/val_endtoend.npy'),
    'rotation':          Approach('Rotation-pretrained encoder', 'results/rotation_encoder.pt',     'results/train_rotation.npy',           None),
    'classify':          Approach('Classify-pretrained encoder', 'results/classify_encoder.pt',     'results/train_classify.npy',           None),
    'frozen_random':     Approach('Frozen random probe',         'results/frozen_random.pt',        'results/train_frozen_random.npy',      'results/val_frozen_random.npy'),
    'frozen_pretrained': Approach('Frozen pretrained probe',     'results/frozen_pretrained.pt',    'results/train_frozen_pretrained.npy',  'results/val_frozen_pretrained.npy'),
    'finetune':          Approach('Finetune pretrained',         'results/finetune.pt',             'results/train_finetune.npy',           'results/val_finetune.npy'),
    'ec_frozen':         Approach('EC: best frozen probe',       'results/ec_frozen.pt',            'results/train_ec_frozen.npy',          'results/val_ec_frozen.npy'),
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
                                 't1_classify',      # Extra credit
                                 't1_ec_pretrain',   # Extra credit
                                 't2_transfer'])
    parser.add_argument('--data', default=os.path.join('..', 'data'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.chdir(sys.path[0])
    os.makedirs('results', exist_ok=True)

    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Task 0: End-to-end scene classification
    #
    if args.task == 't0_endtoend':
        classify_15scenes_data = SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=ENDTOEND_IMAGE_SIZE, batch_size=ENDTOEND_BATCH_SIZE,
        )
        t0_endtoend(classify_15scenes_data, device, APPROACHES)

    # Task 1: Rotation pretraining (1 image)
    #
    elif args.task == 't1_rotation':
        rotation_data = CropRotationDataset(
            os.path.join(args.data, 'single-images', 'train', 'Street'),
            num_crops=ROTATION_NUM_CROPS, crop_size=ROTATION_CROP_SIZE,
            rotation=True, batch_size=ROTATION_BATCH_SIZE,
        )
        t1_rotation(rotation_data, device, APPROACHES)

    # Task 2: Transfer evaluation
    #
    elif args.task == 't2_transfer':
        classify_15scenes_data = SceneDataset(
            os.path.join(args.data, '15-scenes-csci1430'),
            image_size=TRANSFER_IMAGE_SIZE, batch_size=TRANSFER_BATCH_SIZE,
        )
        t2_transfer(classify_15scenes_data, device, APPROACHES)

    # Extra Credit: Classification pretraining (2 images)
    #
    elif args.task == 't1_classify':
        classify_data = CropRotationDataset(
            os.path.join(args.data, 'single-images', 'train'),
            num_crops=CLASSIFY_NUM_CROPS, crop_size=CLASSIFY_CROP_SIZE,
            rotation=False, batch_size=CLASSIFY_BATCH_SIZE,
        )
        t1_classify(classify_data, device, APPROACHES)

    # Extra Credit: Open-ended self-supervised pretraining
    #
    elif args.task == 't1_ec_pretrain':
        t1_ec_pretrain(device, APPROACHES)
