import os
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm
import pyvista as pv
import dbdicom as db
import zarr

import miblab_ssa as ssa


def mosaic_masks_dcm(masks, imagefile, labels=None, view_vector=(1, 0, 0)):

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    plotter = pv.Plotter(window_size=(ncols*width, nrows*height), shape=(nrows, ncols), border=False, off_screen=True)
    plotter.background_color = 'white'

    row = 0
    col = 0
    for i, mask_series in tqdm(enumerate(masks), desc=f'Building mosaic'):

        # Set up plotter
        plotter.subplot(row,col)
        if labels is not None:
            plotter.add_text(labels[i], font_size=6)
        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1

        # Load data
        vol = db.volume(mask_series, verbose=0)
        mask_norm = ssa.sdf_ft.smooth_mask(vol.values, order=32)

        # Plot tile
        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = vol.spacing
        orig_surface = orig_vol.contour(isosurfaces=[0.5])
        plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
        plotter.camera_position = 'iso'
        plotter.view_vector(view_vector)  # rotate 180° around vertical axis
    
    plotter.screenshot(imagefile)
    plotter.close()


def rotating_masks_grid(
        dir_output: str, 
        masks: Union[zarr.Array, np.ndarray], 
        labels: np.ndarray = None,
        nviews = 25,
):
    # 1. Setup metadata
    width, height = 150, 150
    angles = np.linspace(0, 2*np.pi, nviews)
    # View directions (Z-rotation and Y-rotation)
    dirs = [(np.cos(a), np.sin(a), 0.0) for a in angles] 
    dirs += [(np.cos(a), 0.0, np.sin(a)) for a in angles]

    ncols, nrows = masks.shape[0], masks.shape[1]
    os.makedirs(dir_output, exist_ok=True)

    # 2. Outer Loop: Views (The "Memory Clear" loop)
    # By putting 'views' on the outside, we only ever need ONE plotter in RAM.
    for i, vec in enumerate(tqdm(dirs, desc="Rendering View Angles")):
        
        plotter = pv.Plotter(
            window_size=(ncols*width, nrows*height), 
            shape=(nrows, ncols), 
            border=False, 
            off_screen=True,
        )
        plotter.background_color = 'white'

        # 3. Inner Loops: Grid
        for row in range(nrows):
            for col in range(ncols):
                # Load mask (27MB bool) -> Convert to 109MB float32 (not float64!)
                mask_norm = masks[col, row, ...].astype(np.float32)

                # Generate Surface Mesh (The 10x-to-Mesh conversion)
                # 
                vol = pv.wrap(mask_norm)
                vol.spacing = [1.0, 1.0, 1.0]
                surf = vol.contour(isosurfaces=[0.5])
                
                # Cleanup volume from RAM immediately
                del vol

                # Camera Math
                distance = surf.length * 2.5
                center = list(surf.center)
                pos = center + distance * np.array(vec)
                
                # Plotting
                plotter.subplot(row, col)
                if labels is not None:
                    plotter.add_text(str(labels[col, row]), font_size=6, color='black')
                
                plotter.add_mesh(surf, color='lightblue', smooth_shading=True)
                plotter.camera_position = [pos, center, (0, 0, 1)] # Simple Up-vector

        # 4. Save and Destroy
        # This is where the RAM resets to zero for the next angle
        file = os.path.join(dir_output, f"mosaic_{i:03d}.png")
        plotter.screenshot(file)
        plotter.close() # CRITICAL: Releases the GPU and RAM buffers
        del plotter

# def rotating_masks_grid(
#         dir_output:str, 
#         masks:Union[zarr.Array, np.ndarray], 
#         labels:np.ndarray=None,
#         nviews=25,
# ):
#     # masks: (cols, rows) + 3d shape
#     # labels: (cols, rows)
#     # Plot settings
#     width = 150
#     height = 150

#     # Define view points
#     angles = np.linspace(0, 2*np.pi, nviews)
#     dirs = [(np.cos(a), np.sin(a), 0.0) for a in angles] # rotate around z
#     dirs += [(np.cos(a), 0.0, np.sin(a)) for a in angles] # rotate around y

#     # Count nr of mosaics
#     ncols = masks.shape[0]
#     nrows = masks.shape[1]

#     plotters = {}
#     for i, vec in enumerate(dirs):
#         plotters[i] = pv.Plotter(
#             window_size=(ncols*width, nrows*height), 
#             shape=(nrows, ncols), 
#             border=False, 
#             off_screen=True,
#         )
#         plotters[i].background_color = 'white'

#     for row in tqdm(range(nrows), desc=f'Building mosaic'):
#         for col in range(ncols):

#             # Load data once
#             mask_norm = masks[col, row, ...]

#             orig_vol = pv.wrap(mask_norm.astype(float))
#             orig_vol.spacing = [1.0, 1.0, 1.0]
#             orig_surface = orig_vol.contour(isosurfaces=[0.5])

#             prev_up = None
#             for i, vec in enumerate(dirs):
#                 # Camera position
#                 distance = orig_surface.length * 2.0  # controls zoom
#                 center = list(orig_surface.center)
#                 pos = center + distance * np.array(vec) # vec = direction
#                 up = _camera_up_from_direction(vec, prev_up)
#                 prev_up = up

#                 # Set up plotter
#                 plotters[i].subplot(row, col)
#                 if labels is not None:
#                     plotters[i].add_text(labels[col, row], font_size=6)
#                 plotters[i].add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
#                 plotters[i].camera_position = [pos, center, up]

#     for i, vec in tqdm(enumerate(dirs), desc='Saving mosaics..'):
#         file = os.path.join(dir_output, f"mosaic_{i:03d}.png")
#         os.makedirs(Path(file).parent, exist_ok=True)
#         plotters[i].screenshot(file)
#         plotters[i].close()



def rotating_mosaics_da(dir_output, masks:list, labels=None, chunksize=None, nviews=25, columns=None, rows=None):
    # masks - list of numpy or dask arrays
    if labels is None:
        labels = [str(i) for i in range(len(masks))]
    if chunksize is None:
        chunksize = len(masks)

    # Split into numbered chunks
    def chunk_list(lst, size):
        chunks = [lst[i:i+size] for i in range(0, len(lst), size)]
        return list(enumerate(chunks))
     
    mask_chunks = chunk_list(masks, chunksize)
    label_chunks = chunk_list(labels, chunksize)

    # Define view points
    angles = np.linspace(0, 2*np.pi, nviews)
    dirs = [(np.cos(a), np.sin(a), 0.0) for a in angles] # rotate around z
    dirs += [(np.cos(a), 0.0, np.sin(a)) for a in angles] # rotate around y

    # Save mosaics for each chunk and view
    for mask_chunk, label_chunk in tqdm(zip(mask_chunks, label_chunks), total=len(label_chunks), desc='Building chunk of mosaics'):
        chunk_idx = mask_chunk[0]
        names = [f"group_{str(chunk_idx).zfill(2)}_{i:02d}.png" for i in range(len(dirs))]
        directions = {vec: os.path.join(dir_output, name) for name, vec in zip(names, dirs)}
        multiple_mosaic_masks_da(mask_chunk[1], directions, label_chunk[1], columns=columns, rows=rows)
        

def multiple_mosaic_masks_da(masks:list, directions:dict, labels, columns=None, rows=None):
    # masks - list of dask arrays
    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    if columns is None:
        ncols = int(np.ceil(np.sqrt((height*n_mosaics)/(aspect_ratio*width))))
    else:
        ncols = columns
    if rows is None:
        nrows = int(np.ceil(n_mosaics/ncols))
    else:
        nrows = rows
    # nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    # ncols = int(np.ceil(n_mosaics/nrows))

    plotters = {}
    for vec in directions.keys():
        plotters[vec] = pv.Plotter(
            window_size=(ncols*width, nrows*height), 
            shape=(nrows, ncols), 
            border=False, 
            off_screen=True,
        )
        plotters[vec].background_color = 'white'

    row = 0
    col = 0
    for mask_label, mask_series in tqdm(zip(labels, masks), total=len(labels), desc=f'Building mosaic'):

        # Load data once
        mask_norm = ssa.sdf_ft.smooth_mask(mask_series[:].astype(bool), order=16)

        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = [1.0, 1.0, 1.0]
        orig_surface = orig_vol.contour(isosurfaces=[0.5])

        prev_up = None
        for vec in directions.keys():
            # Camera position
            distance = orig_surface.length * 2.0  # controls zoom
            center = list(orig_surface.center)
            pos = center + distance * np.array(vec) # vec = direction
            up = _camera_up_from_direction(vec, prev_up)
            prev_up = up

            # Set up plotter
            plotter = plotters[vec]
            plotter.subplot(row,col)
            if labels is not None:
                plotter.add_text(mask_label, font_size=6)
            plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
            plotter.camera_position = [pos, center, up]
            
            # plotter.camera_position = 'iso'
            # plotter.view_vector(vec)  # rotate 180° around vertical axis

        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1
    
    for vec, file in directions.items():
        # plotters[vec].render()
        plotters[vec].screenshot(file)
        plotters[vec].close()


def rotating_mosaics_npz(dir_output, masks:list, labels=None, chunksize=None, nviews=25, columns=None, rows=None):

    if labels is None:
        labels = [str(i) for i in range(len(masks))]
    if chunksize is None:
        chunksize = len(masks)

    # Split into numbered chunks
    def chunk_list(lst, size):
        chunks = [lst[i:i+size] for i in range(0, len(lst), size)]
        return list(enumerate(chunks))
     
    mask_chunks = chunk_list(masks, chunksize)
    label_chunks = chunk_list(labels, chunksize)

    # Define view points
    angles = np.linspace(0, 2*np.pi, nviews)
    dirs = [(np.cos(a), np.sin(a), 0.0) for a in angles] # rotate around z
    dirs += [(np.cos(a), 0.0, np.sin(a)) for a in angles] # rotate around y

    # Save mosaics for each chunk and view
    for mask_chunk, label_chunk in zip(mask_chunks, label_chunks):
        chunk_idx = mask_chunk[0]
        names = [f"group_{str(chunk_idx).zfill(2)}_{i:02d}.png" for i in range(len(dirs))]
        directions = {vec: os.path.join(dir_output, name) for name, vec in zip(names, dirs)}
        multiple_mosaic_masks_npz(mask_chunk[1], directions, label_chunk[1], columns=columns, rows=rows)
        

def multiple_mosaic_masks_npz(masks, directions:dict, labels, columns=None, rows=None):
    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    if columns is None:
        ncols = int(np.ceil(np.sqrt((height*n_mosaics)/(aspect_ratio*width))))
    else:
        ncols = columns
    if rows is None:
        nrows = int(np.ceil(n_mosaics/ncols))
    else:
        nrows = rows
    # nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    # ncols = int(np.ceil(n_mosaics/nrows))

    plotters = {}
    for vec in directions.keys():
        plotters[vec] = pv.Plotter(
            window_size=(ncols*width, nrows*height), 
            shape=(nrows, ncols), 
            border=False, 
            off_screen=True,
        )
        plotters[vec].background_color = 'white'

    row = 0
    col = 0
    for mask_label, mask_series in tqdm(zip(labels, masks), desc=f'Building mosaic'):

        # Load data once
        vol = db.npz.volume(mask_series)
        mask_norm = ssa.sdf_ft.smooth_mask(vol.values.astype(bool), order=32)

        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = [1.0, 1.0, 1.0]
        orig_surface = orig_vol.contour(isosurfaces=[0.5])

        prev_up = None
        for vec in directions.keys():
            # Camera position
            distance = orig_surface.length * 2.0  # controls zoom
            center = list(orig_surface.center)
            pos = center + distance * np.array(vec) # vec = direction
            up = _camera_up_from_direction(vec, prev_up)
            prev_up = up

            # Set up plotter
            plotter = plotters[vec]
            plotter.subplot(row,col)
            if labels is not None:
                plotter.add_text(mask_label, font_size=6)
            plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
            plotter.camera_position = [pos, center, up]
            
            # plotter.camera_position = 'iso'
            # plotter.view_vector(vec)  # rotate 180° around vertical axis

        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1
    
    for vec, file in directions.items():
        # plotters[vec].render()
        plotters[vec].screenshot(file)
        plotters[vec].close()



def _camera_up_from_direction(d, prev_up=None):
    d = np.asarray(d, float)
    d /= np.linalg.norm(d)

    # 1. First Frame: Use your original logic to establish an initial Up vector
    if prev_up is None:
        ref = np.array([0, 0, 1])
        # If looking straight down Z, switch ref to Y to avoid singularity
        if abs(np.dot(d, ref)) > 0.99:
            ref = np.array([0, 1, 0])
        
        right = np.cross(ref, d)
        right /= np.linalg.norm(right)
        up = np.cross(d, right)
    
    # 2. Subsequent Frames: Parallel Transport
    else:
        # Project the previous Up vector onto the plane perpendicular to the new direction.
        # This removes the component of prev_up that is parallel to d.
        # Formula: v_perp = v - (v . d) * d
        up = prev_up - np.dot(prev_up, d) * d
        
        # Normalize the result
        norm = np.linalg.norm(up)
        
        # Handle rare edge case where d aligns perfectly with prev_up (norm is 0)
        if norm < 1e-6:
            # Fallback to initial logic
            ref = np.array([0, 0, 1])
            if abs(np.dot(d, ref)) > 0.99:
                ref = np.array([0, 1, 0])
            right = np.cross(ref, d)
            up = np.cross(d, right)
            up /= np.linalg.norm(up)
        else:
            up /= norm

    return up


def mosaic_masks_npz(masks, imagefile, labels=None, view_vector=(1, 0, 0)):

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    plotter = pv.Plotter(
        window_size=(ncols*width, nrows*height), 
        shape=(nrows, ncols), 
        border=False, 
        off_screen=True,
    )
    plotter.background_color = 'white'

    row = 0
    col = 0
    for i, mask_series in tqdm(enumerate(masks), desc=f'Building mosaic'):

        # Set up plotter
        plotter.subplot(row,col)
        if labels is not None:
            plotter.add_text(labels[i], font_size=6)
        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1

        # Load data
        vol = db.npz.volume(mask_series)
        mask_norm = ssa.sdf_ft.smooth_mask(vol.values.astype(bool), order=32)

        # Plot tile
        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = [1.0, 1.0, 1.0]
        orig_surface = orig_vol.contour(isosurfaces=[0.5])
        plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
        plotter.camera_position = 'iso'
        plotter.view_vector(view_vector)  # rotate 180° around vertical axis
    
    plotter.screenshot(imagefile)
    plotter.close()


def mosaic_features_npz(features, imagefile, labels=None, view_vector=(1, 0, 0)):

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(features)
    nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    plotter = pv.Plotter(
        window_size=(ncols*width, nrows*height), 
        shape=(nrows, ncols), 
        border=False, 
        off_screen=True,
    )
    plotter.background_color = 'white'

    row = 0
    col = 0
    for i, feat in tqdm(enumerate(features), desc=f'Building mosaic'):

        # Set up plotter
        plotter.subplot(row,col)
        if labels is not None:
            plotter.add_text(labels[i], font_size=6)
        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1

        # Load data
        ft = np.load(feat)
        mask_norm = ssa.sdf_ft.mask_from_features(ft['features'], ft['shape'], ft['order'])

        # Plot tile
        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = [1.0, 1.0, 1.0]
        orig_surface = orig_vol.contour(isosurfaces=[0.5])
        plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
        plotter.camera_position = 'iso'
        plotter.view_vector(view_vector)  # rotate 180° around vertical axis
    
    plotter.screenshot(imagefile)
    plotter.close()