import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def save_video_as_mp4(video_tensor, filename, fps=25):
    # Convert the tensor to a numpy array in the channel-last format
    video_np = video_tensor.permute(0, 2, 3, 1).numpy()

    # Convert the numpy array to RGB format
    video_rgb = np.flip(video_np, axis=-1)

    # Set up the video writer
    writer = FFMPEG_VideoWriter(filename, (video_rgb.shape[2], video_rgb.shape[1]), fps=fps)

    # Write each frame to the video file
    for frame in video_rgb:
        writer.write_frame(frame)

    # Close the video writer
    writer.close()
