"""
Homework 4 - CNNs: Pretraining an Encoder
CSCI1430 - Computer Vision
Brown University
"""
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

# ========================================================================
#  Visualization
# ========================================================================

def visualize_filters(model, save_path=None):
    """Extract and display the first conv layer's learned filters."""

    conv1 = model.layers[0]
    weights = conv1.weight.data.cpu()

    n_filters = weights.shape[0]
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            w = weights[i]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(w.permute(1, 2, 0).numpy())
        ax.axis('off')

    plt.suptitle('Conv1 Filters')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def save_filter_frame(encoder, epoch, output_dir='filter_frames'):
    """Save one frame of conv1 filter visualization for the given epoch.

    Call this from an on_epoch_end callback during training.
    After training, call make_filter_video() to assemble the frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    conv1 = encoder.layers[0]
    weights = conv1.weight.data.cpu()

    n_filters = weights.shape[0]
    cols = 8
    rows = (n_filters + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            w = weights[i]
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            ax.imshow(w.permute(1, 2, 0).numpy())
        ax.axis('off')

    plt.suptitle(f'Conv1 Filters \u2014 Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch:03d}.png'),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


def make_filter_video(frame_dir, output_path='filters.gif', duration_ms=200):
    """Assemble saved filter frames into an animated GIF.

    Args:
        frame_dir:   directory containing epoch_000.png, epoch_001.png, ...
        output_path: where to save the GIF
        duration_ms: milliseconds per frame (200 = 5 fps)
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, 'epoch_*.png')))
    if not paths:
        print(f"No frames found in {frame_dir}")
        return

    frames = [Image.open(p) for p in paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"Saved {len(frames)}-frame animation -> {output_path}")
