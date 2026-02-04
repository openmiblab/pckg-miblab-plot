import os

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # NEW (v2.x)

def images_to_video(image_folder, output_file, fps=30):
    # 1. Get and sort images
    images = [os.path.join(image_folder, img) 
              for img in os.listdir(image_folder) 
              if img.endswith(".png")]
    images.sort()

    if not images:
        print("No images found!")
        return

    # 2. Create the clip
    clip = ImageSequenceClip(images, fps=fps)
    
    # 3. Write to MP4 (Windows compatible)
    # The 'logger=None' argument suppresses the progress bar if you want cleaner output
    clip.write_videofile(output_file, codec='libx264', bitrate="50000k")

# Usage
# save_high_quality_mp4('your/image/folder', 'video.mp4')

# Usage
# create_lossless_video('your_folder', 'high_quality.mp4')

# Usage
# save_high_quality_mp4('my_folder', 'final_video.mp4')

# def _images_to_video(image_folder, output_video_file, fps=30):
#     """
#     Converts a folder of PNG images into a video file.
    
#     Args:
#         image_folder (str): Path to the folder containing images.
#         output_video_file (str): Output filename (e.g., 'output.mp4').
#         fps (int): Frames per second.
#     """
    
#     # 1. Get the list of files
#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
#     # 2. Sort the images to ensure they are in the correct order
#     # Note: This uses standard string sorting. If your files are named 1.png, 10.png, 2.png,
#     # you might need "natural sorting" logic.
#     images.sort()

#     if not images:
#         print("No PNG images found in the directory.")
#         return

#     # 3. Read the first image to determine width and height
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape
#     size = (width, height)

#     # 4. Define the codec and create VideoWriter object
#     # 'mp4v' is a standard codec for MP4 containers
#     # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Motion JPEG codec
#     out = cv2.VideoWriter(output_video_file, fourcc, fps, size)
    

#     print(f"Processing {len(images)} images...")

#     # 5. Write images to video
#     for image in images:
#         img_path = os.path.join(image_folder, image)
#         frame = cv2.imread(img_path)
        
#         # specific check: resizing might be needed if images vary in size
#         # frame = cv2.resize(frame, size) 
        
#         out.write(frame)

#     # 6. Release everything
#     out.release()
#     print(f"Video saved as {output_video_file}")

# # --- Usage Example ---
# # images_to_video('path/to/your/images', 'my_animation.mp4', fps=24)