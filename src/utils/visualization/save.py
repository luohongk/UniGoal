import os
import cv2


def save_video(vis_image_list, video_path):
    save_video_dir = os.path.dirname(video_path)
    os.makedirs(save_video_dir, exist_ok=True)
    
    frame = vis_image_list[0]
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 4.0, (width // 2, height // 2))
    
    for image in vis_image_list:
        image = cv2.resize(image, (width // 2, height // 2))
        video.write(image)
    
    video.release()
    vis_image_list = []