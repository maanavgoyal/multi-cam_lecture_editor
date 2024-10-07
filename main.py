import os
import torch
from src.video_processor import load_video
from src.frame_selector import FrameSelector
from src.video_editor import create_final_video
from src.video_synchronizer import synchronize_videos
from models.video_swin_transformer import VideoSwinTransformer

def main():
    # Load Video Swin Transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoSwinTransformer().to(device)
    model.eval()
    
    # Load input videos
    input_dir = 'data/input_videos'
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    # Synchronize videos
    print("Synchronizing videos...")
    offsets = synchronize_videos(model, video_paths)
    print(f"Computed offsets: {offsets}")
    
    # Load synchronized video frames
    all_frames = []
    for path, offset in zip(video_paths, offsets):
        all_frames.append(load_video(path, start_frame=max(0, offset)))
    
    # Adjust for negative offsets
    min_offset = min(offsets)
    if min_offset < 0:
        for i in range(len(all_frames)):
            all_frames[i] = all_frames[i][abs(min_offset):]
    
    # Ensure all videos have the same number of frames
    min_frames = min(len(frames) for frames in all_frames)
    all_frames = [frames[:min_frames] for frames in all_frames]

    # Initialize frame selector
    frame_selector = FrameSelector(len(video_paths))
    
    # Select best frames
    selected_frames = []
    for frame_set in zip(*all_frames):
        best_frame = frame_selector.select_best_frame(frame_set)
        selected_frames.append(best_frame)
    
    # Create final video
    create_final_video(selected_frames, 'output/final_lecture.mp4')

if __name__ == "__main__":
    main()