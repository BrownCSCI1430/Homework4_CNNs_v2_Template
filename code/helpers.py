"""
Homework 4 - CNNs: Pretraining an Encoder
CSCI1430 - Computer Vision
Brown University
"""
import os
import glob
import subprocess
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


def save_filter_frame(encoder, epoch, output_dir='results/filter_frames'):
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


def make_filter_video(frame_dir, output_path='filters.mp4', fps=5):
    """Assemble saved filter frames into a video (MP4 via ffmpeg, GIF fallback).

    Args:
        frame_dir:   directory containing epoch_000.png, epoch_001.png, ...
        output_path: where to save the video (.mp4 or .gif)
        fps:         frames per second (default 5)
    """
    paths = sorted(glob.glob(os.path.join(frame_dir, 'epoch_*.png')))
    if not paths:
        print(f"No frames found in {frame_dir}")
        return

    # Force .mp4 extension
    if output_path.endswith('.gif'):
        output_path = output_path[:-4] + '.mp4'

    # Try ffmpeg first (available on CCV cluster nodes)
    pattern = os.path.join(frame_dir, 'epoch_%03d.png')
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', pattern,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path,
        ], check=True, capture_output=True)
        print(f"Saved {len(paths)}-frame video -> {output_path}")
        return
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ffmpeg unavailable ({e}), falling back to GIF")

    # Fallback: GIF via PIL
    gif_path = output_path.replace('.mp4', '.gif')
    frames = [Image.open(p) for p in paths]
    duration_ms = 1000 // fps
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    print(f"Saved {len(frames)}-frame animation -> {gif_path}")


def _conv1_diagnostics(encoder, w0, w_prev, epoch, frame_dir):
    """Print conv1 weight diagnostics and save filter frames (raw + delta)."""
    w = encoder.layers[0].weight.data.cpu()
    diff_from_init = (w - w0).abs().mean().item()
    diff_from_prev = (w - w_prev[0]).abs().mean().item()
    mag = w.abs().mean().item()
    w_std = w.std().item()

    grad_str = ""
    if encoder.layers[0].weight.grad is not None:
        grad_norm = encoder.layers[0].weight.grad.data.abs().mean().item()
        grad_str = f"  grad={grad_norm:.6f}"

    print(f"  conv1: diff_init={diff_from_init:.4f}  diff_ep={diff_from_prev:.5f}  "
          f"|w|={mag:.4f}  std={w_std:.4f}  ratio={diff_from_init/mag:.0%}{grad_str}",
          flush=True)

    save_filter_frame(encoder, epoch, output_dir=frame_dir)

    # Save delta filter frame (w - w0) — shows what the network learned
    delta_dir = frame_dir + '_delta'
    os.makedirs(delta_dir, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    delta = w - w0
    n = delta.shape[0]
    cols, rows = 8, (n + 7) // 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            f = delta[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f.permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.suptitle(f'Learned Delta (w - w0) -- Epoch {epoch + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(delta_dir, f'epoch_{epoch:03d}.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)

    w_prev[0] = w.clone()


def make_filter_callback(encoder, frame_dir, filter_save_path):
    """Create an on_epoch_end callback for filter visualization.

    Usage:
        callback = make_filter_callback(encoder, 'results/filter_frames_rotation',
                                        'results/conv1_filters_rotation.png')
        train_loop(..., on_epoch_end=callback)
    """
    w0 = encoder.layers[0].weight.data.cpu().clone()
    w_prev = [w0.clone()]
    def callback(epoch, model):
        _conv1_diagnostics(encoder, w0, w_prev, epoch, frame_dir)
        visualize_filters(encoder, save_path=filter_save_path)
    return callback
