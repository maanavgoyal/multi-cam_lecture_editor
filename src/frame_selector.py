import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_motion(frames):
    motion = 0
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion += np.sum(np.abs(flow))
    return motion

def detect_slides(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.sum(edges) > 50000  # Adjust threshold as needed

def measure_slide_clarity(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.var(laplacian)

class FrameSelector:
    def __init__(self, num_videos, switch_threshold=5):
        self.num_videos = num_videos
        self.switch_threshold = switch_threshold
        self.current_video = 0
        self.frames_since_switch = 0

    def select_best_frame(self, frames):
        motions = [compute_motion(video_frames) for video_frames in frames]
        slide_scores = [measure_slide_clarity(video_frames[-1]) if detect_slides(video_frames[-1]) else 0 for video_frames in frames]
        
        # Prioritize slides if present
        max_slide_score = max(slide_scores)
        if max_slide_score > 0:
            best_video = slide_scores.index(max_slide_score)
        else:
            best_video = motions.index(max(motions))
        
        # Apply switching heuristic
        if best_video != self.current_video:
            if self.frames_since_switch < self.switch_threshold:
                best_video = self.current_video
            else:
                self.current_video = best_video
                self.frames_since_switch = 0
        else:
            self.frames_since_switch += 1
        
        return frames[best_video][-1]