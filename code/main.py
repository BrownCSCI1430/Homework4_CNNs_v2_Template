"""
Homework 4 - CNNs
CSCI1430 - Computer Vision
Brown University

What happens when we train encoder architectures with different
data and supervision? This homework explores how these choices
affect the features a CNN learns for a downstream task.

    uv run python main.py --task <task>

Task:   Args:                           Description:
 0      -                               Set up your data loaders
 1      t1_rotation_1img                Train filters with self-supervised rotation.
                                        One image dataset 'Ameyoko' street scene only!
 2      t2_classify_2img                Train filters with supervised labels.
                                        Two images dataset 'Ameyoko/Squeeky Beach' only (one per class).
 3      t3_pretrained_classify_15scenes Use filters from T1, T2 to train a classification head
                                        on the 15-scenes dataset.
 4      t4_endtoend_classify_15scenes   Design your own SceneClassifier and train it
                                        end-to-end on the 15-scenes dataset.
 5      t5_compare                      Compare the performance of features learned in each!
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
    t1_rotation_1img, t2_classify_2img,
    t3_pretrained_classify_15scenes, t4_endtoend_classify_15scenes,
)
from hyperparameters import *

# ========================================================================
#  Task 5: Compare — how do different initializations affect learning?
#
#  Load the learning curves saved by Tasks 3 & 4 and plot them side by side.
# ========================================================================

def t5_compare():
    # Load saved curves from Tasks 3 & 4
    fig, ax = plt.subplots(figsize=(8, 5))
    for approach in APPROACHES.values():
        accs = np.load(approach.curve)
        ax.plot(range(1, len(accs) + 1), accs, label=approach.label)
        print(f"Loaded {approach.curve}: {len(accs)} epochs, best val {max(accs):.3f}")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.legend()
    ax.set_title('Effect of Pretraining on Downstream Classification')
    plt.tight_layout()
    plt.savefig('pretrain_comparison.png', dpi=150)
    print("Saved pretrain_comparison.png")


# ========================================================================
#  Dispatch plumbing
# ========================================================================

Approach = namedtuple('Approach', ['label', 'weights', 'curve'])

APPROACHES = {
    'random':    Approach('Random features',                            None,                            'curve_random.npy'),
    'rotation':  Approach('Pretrained rotation task (1 image)',         'rotation_1img_encoder.pt',      'curve_rotation_1img_encoder.npy'),
    'classify':  Approach('Pretrained classify task (2 images)',        'classify_2img_encoder.pt',      'curve_classify_2img_encoder.npy'),
    'endtoend':  Approach('End-to-end (1,500 images SceneClassifier)',  'endtoend_classify_15scenes.pt', 'curve_endtoend_classify.npy'),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4: Pretraining an Encoder")
    parser.add_argument('--task', required=True,
                        choices=['t1_rotation_1img', 't2_classify_2img',
                                 't3_pretrained_classify_15scenes',
                                 't4_endtoend_classify_15scenes', 't5_compare'])
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

    # Task 0: Load all datasets
    #
    # Our main task dataset
    classify_15scenes_data = SceneDataset(
        os.path.join(args.data, '15-scenes-csci1430'), image_size=IMAGE_SIZE,
    )

    # Our rotation self-supervised dataset for a single Ameyoko street scene
    rotation_1img_data = CropRotationDataset(
        os.path.join(args.data, 'single-images', 'train', 'Street'),
        num_crops=NUM_CROPS, crop_size=IMAGE_SIZE, rotation=True,
        batch_size=BATCH_SIZE,
    )

    # Our classification task but just with two images:
    # Ameyoko (Street class) and Squeeky Beach (Coast class)
    classify_2img_data = CropRotationDataset(
        os.path.join(args.data, 'single-images', 'train'),
        num_crops=NUM_CROPS, crop_size=IMAGE_SIZE, rotation=False,
        batch_size=BATCH_SIZE,
    )

    # Tasks 1-5: Execute the selected task
    if args.task == 't1_rotation_1img':
        t1_rotation_1img(rotation_1img_data, device, APPROACHES)
    elif args.task == 't2_classify_2img':
        t2_classify_2img(classify_2img_data, device, APPROACHES)
    elif args.task == 't3_pretrained_classify_15scenes':
        t3_pretrained_classify_15scenes(classify_15scenes_data, device, APPROACHES)
    elif args.task == 't4_endtoend_classify_15scenes':
        t4_endtoend_classify_15scenes(classify_15scenes_data, device, APPROACHES)
    elif args.task == 't5_compare':
        t5_compare()
