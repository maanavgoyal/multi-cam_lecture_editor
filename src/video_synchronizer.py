import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def load_video_segment(video_path, start_frame, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames):
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(frame) for frame in frames]).unsqueeze(0)

def extract_features(model, frames):
    with torch.no_grad():
        features = model(frames)
    return features.squeeze().cpu().numpy()

def compute_similarity(features1, features2):
    return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))

def find_sync_point(model, reference_video, target_video, max_offset=300):
    best_offset = 0
    best_similarity = -1

    for offset in range(-max_offset, max_offset + 1, 16):
        ref_frames = load_video_segment(reference_video, max(0, -offset))
        target_frames = load_video_segment(target_video, max(0, offset))

        if len(ref_frames) < 16 or len(target_frames) < 16:
            continue

        ref_input = preprocess_frames(ref_frames)
        target_input = preprocess_frames(target_frames)

        ref_features = extract_features(model, ref_input)
        target_features = extract_features(model, target_input)

        similarity = compute_similarity(ref_features, target_features)

        if similarity > best_similarity:
            best_similarity = similarity
            best_offset = offset

    return best_offset

def synchronize_videos(model, video_paths):
    reference_video = video_paths[0]
    offsets = [0]  # Reference video has 0 offset
    for video_path in video_paths[1:]:
        offset = find_sync_point(model, reference_video, video_path)
        offsets.append(offset)
    return offsets