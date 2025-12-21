from math import ceil

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm



def get_distinct_colors(rois, colormap='jet', opacity=0.6):
    if len(rois)==1:
        colors = [[1, 0, 0, opacity]]
    elif len(rois)==2:
        colors = [[1, 0, 0, opacity], [0, 1, 0, opacity]]
    elif len(rois)==3:
        colors = [[1, 0, 0, opacity], [0, 1, 0, opacity], [0, 0, 1, opacity]]
    else:
        n = len(rois)
        #cmap = cm.get_cmap(colormap, n)
        cmap = matplotlib.colormaps[colormap]
        colors = [cmap(i)[:3] + (opacity,) for i in np.linspace(0, 1, n)]

    return colors


def mosaic_overlay(
        img, 
        rois, 
        file, 
        colormap='tab20', 
        aspect_ratio=16/9, 
        margin=None,
        vmin=None,
        vmax=None,
        opacity=0.6,
    ):

    # Set defaults color window
    if vmin is None:
        vmin=0
    if vmax is None:
        vmax=np.mean(img) + 2 * np.std(img)

    # Define RGBA colors (R, G, B, Alpha) — alpha controls transparency
    colors = get_distinct_colors(rois, colormap=colormap, opacity=opacity)

    # Get all masks as boolean arrays
    masks = [m.astype(bool) for m in rois.values()]

    # Build a single combined mask
    all_masks = masks[0]
    for i in range(1, len(masks)):
        all_masks = np.logical_or(all_masks, masks[i])
    if np.sum(all_masks)==0:
        raise ValueError('Empty masks')
    
    # Find corners of cropped mask
    for x0 in range(all_masks.shape[0]):
        if np.sum(all_masks[x0,:,:]) > 0:
            break
    for x1 in range(all_masks.shape[0]-1, -1, -1):
        if np.sum(all_masks[x1,:,:]) > 0:
            break
    for y0 in range(all_masks.shape[1]):
        if np.sum(all_masks[:,y0,:]) > 0:
            break
    for y1 in range(all_masks.shape[1]-1, -1, -1):
        if np.sum(all_masks[:,y1,:]) > 0:
            break
    for z0 in range(all_masks.shape[2]):
        if np.sum(all_masks[:,:,z0]) > 0:
            break
    for z1 in range(all_masks.shape[2]-1, -1, -1):
        if np.sum(all_masks[:,:,z1]) > 0:
            break

    # Add in the margins     
    if margin is None:
        x0 = 0
        y0 = 0
        z0 = 0
        x1 = all_masks.shape[0] - 1
        y1 = all_masks.shape[1] - 1
        z1 = all_masks.shape[2] - 1
    else:
        x0 = x0-margin[0] if x0-margin[0]>=0 else 0
        y0 = y0-margin[1] if y0-margin[1]>=0 else 0
        z0 = z0-margin[2] if z0-margin[2]>=0 else 0
        x1 = x1+margin[0] if x1+margin[0]<all_masks.shape[0] else all_masks.shape[0]-1
        y1 = y1+margin[1] if y1+margin[1]<all_masks.shape[1] else all_masks.shape[1]-1
        z1 = z1+margin[2] if z1+margin[2]<all_masks.shape[2] else all_masks.shape[2]-1

    # Determine number of rows and columns
    # c*r = n -> c=n/r
    # c*w / r*h = a -> w*n/r = a*r*h -> (w*n) / (a*h) = r**2
    width = x1-x0+1
    height = y1-y0+1
    n_mosaics = z1-z0+1
    nrows = int(np.round(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    # Set up figure 
    fig, ax = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        gridspec_kw = {'wspace':0, 'hspace':0}, 
        figsize=(ncols*width/max([width,height]), nrows*height/max([width,height])),
        dpi=300,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Build figure
    i = 0
    for row in tqdm(ax, desc='Building png'):
        for col in row:

            col.set_xticklabels([])
            col.set_yticklabels([])
            col.set_aspect('equal')
            col.axis("off")

            # Display the background image
            if z0+i < img.shape[2]:
                col.imshow(
                    img[x0:x1+1, y0:y1+1, z0+i].T, 
                    cmap='gray', 
                    interpolation='none', 
                    vmin=vmin, 
                    vmax=vmax,
                )

            # Overlay each mask
            if z0+i <= z1:
                for mask, color in zip(masks, colors):
                    rgba = np.zeros((x1+1-x0, y1+1-y0, 4), dtype=float)
                    for c in range(4):  # RGBA
                        rgba[..., c] = mask[x0:x1+1, y0:y1+1, z0+i] * color[c]
                    col.imshow(rgba.transpose((1,0,2)), interpolation='none')

            i += 1

    # fig.suptitle('Mask overlay', fontsize=14)
    fig.savefig(file, bbox_inches='tight', pad_inches=0)
    plt.close()



def volume_to_mosaic(
        data: np.ndarray,
        axis: int = 2,
        fill_value: int = 0,
        target_ratio = 16.0 / 9.0,
        save_as = None,
        clip = None,
    ) -> np.ndarray:

    """
    Turn a 3D volume into a 2D mosaic that best matches 16:9 aspect ratio.

    Parameters
    ----------
    data : np.ndarray
        3D numpy array.
    fill_value : int, optional
        Pixel value (0-255) to use for padded (empty) tiles if n_slices does not fill grid.
    target_ratio: float
        Width-to-hight ratio of the moisaic

    Returns
    -------
    np.ndarray
        2D numpy array.
    """

    if data.ndim != 3:
        raise ValueError(f"Expected 3D image (or 4D with vol), got shape {data.shape}")
    
    data = data.swapaxes(0,1)

    # reorder so slices are along axis 0 for convenience
    if axis != 0:
        data = np.moveaxis(data, axis, 0)  # now data.shape = (nslices, H, W)

    n_slices, H, W = data.shape

    # --- choose grid (cols x rows) to best match 16:9 ---
    # each tile aspect = W/H. mosaic ratio = (cols * W) / (rows * H) = (cols/rows) * (W/H)
    # so desired cols/rows = target_ratio * (H/W)
    desired_cr = target_ratio * (H / W)
    best = None  # (error, cols, rows)
    for cols in range(1, n_slices + 1):
        rows = ceil(n_slices / cols)
        cr = cols / rows
        err = abs(cr - desired_cr)
        if best is None or err < best[0]:
            best = (err, cols, rows)
    _, cols, rows = best

    # --- create canvas for tiles (before final resize) ---
    tile_w, tile_h = W, H
    canvas_w = cols * tile_w
    canvas_h = rows * tile_h
    canvas = np.full((canvas_h, canvas_w), fill_value, dtype=data.dtype)

    # fill tiles row-major
    slice_idx = 0
    for r in range(rows):
        for c in range(cols):
            if slice_idx < n_slices:
                tile = data[slice_idx]
                y0 = r * tile_h
                x0 = c * tile_w
                canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile
            slice_idx += 1

    if clip is not None:
        canvas = np.clip(canvas, clip[0], clip[1])

    if save_as is not None:
        canvas_norm = (canvas - canvas.min()) / (canvas.max() - canvas.min())
        canvas = (canvas_norm * 255).astype(np.uint8)
        Image.fromarray(canvas).save(save_as)

    return canvas