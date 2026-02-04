import os
import shutil

import imageio.v2 as imageio  # Use v2 interface for compatibility
from moviepy import VideoFileClip
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def get_distinct_colors(rois, colormap='jet'):
    if len(rois)==1:
        colors = [[255, 0, 0, 0.6]]
    elif len(rois)==2:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6]]
    elif len(rois)==3:
        colors = [[255, 0, 0, 0.6], [0, 255, 0, 0.6], [0, 0, 255, 0.6]]
    else:
        n = len(rois)
        #cmap = cm.get_cmap(colormap, n)
        cmap = matplotlib.colormaps[colormap]
        colors = [cmap(i)[:3] + (0.6,) for i in np.linspace(0, 1, n)]  # Set alpha to 0.6 for transparency

    return colors


def movie_overlay(img, rois, file):

    # Define RGBA colors (R, G, B, Alpha) — alpha controls transparency
    colors = get_distinct_colors(rois, colormap='tab20')

    # Directory to store temporary frames
    tmp = os.path.join(os.getcwd(), 'tmp')
    os.makedirs(tmp, exist_ok=True)
    filenames = []

    # Generate and save a sequence of plots
    for i in tqdm(range(img.shape[2]), desc='Building animation..'):

        # Set up figure
        fig, ax = plt.subplots(
            figsize=(5, 5),
            dpi=300,
        )

        # Display the background image
        ax.imshow(img[:,:,i].T, cmap='gray', interpolation='none', vmin=0, vmax=np.mean(img) + 2 * np.std(img))

        # Overlay each mask
        for mask, color in zip([m.astype(bool) for m in rois.values()], colors):
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=float)
            for c in range(4):  # RGBA
                rgba[..., c] = mask[:,:,i] * color[c]
            ax.imshow(rgba.transpose((1,0,2)), interpolation='none')

        # Save eachg image to a tmp file
        fname = os.path.join(tmp, f'frame_{i}.png')
        fig.savefig(fname)
        filenames.append(fname)
        plt.close(fig)

    # Create GIF
    print('Creating movie')
    gif = os.path.join(tmp, 'movie.gif')
    with imageio.get_writer(gif, mode="I", duration=0.2) as writer:
        for fname in filenames:
            image = imageio.imread(fname)
            writer.append_data(image)

    # Load gif
    clip = VideoFileClip(gif)

    # Save as MP4
    clip.write_videofile(file, codec='libx264')

    # Clean up temporary files
    shutil.rmtree(tmp)