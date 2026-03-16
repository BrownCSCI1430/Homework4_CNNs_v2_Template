#!/bin/bash

# ============================================================
# Homework 4 - Learning Visual Features with CNNs
# CSCI1430 - Computer Vision
# Brown University
# SLURM job script
#
# Usage:
#   sbatch slurm_train.sh t0_endtoend    # runs Task 0
#   sbatch slurm_train.sh t1_rotation    # runs Task 1
#   sbatch slurm_train.sh t2_transfer    # runs Task 2
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J hw4_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Default task if none provided as argument
TASK=${1:-t0_endtoend}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "============================================"

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Run training
uv run python main.py --task "$TASK"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
