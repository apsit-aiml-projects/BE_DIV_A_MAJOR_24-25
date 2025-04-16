from ultralytics import YOLO 
import cv2
import pickle
import os
import numpy as np
from utils import get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def is_inside_court(self, foot_position, court_keypoints):
        """Check if a player's foot position is inside the court boundaries."""
        court_polygon = np.array([
            (court_keypoints[0], court_keypoints[1]),  # Top-left
            (court_keypoints[2], court_keypoints[3]),  # Top-right
            (court_keypoints[6], court_keypoints[7]),  # Bottom-right
            (court_keypoints[4], court_keypoints[5])   # Bottom-left
        ], dtype=np.int32)

        return cv2.pointPolygonTest(court_polygon, foot_position, False) >= 0

    def detect_players(self, frames, read_from_stub=False, stub_path=None):
        """Detect all players in video frames and return tracking data."""
        player_detections = []

        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        """Detect players in a single frame using YOLO."""
        results = self.model.track(frame, persist=True)[0]
        player_dict = {}

        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                player_dict[track_id] = bbox  # Store detected players

        return player_dict
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect players in all frames and return tracking data."""
        player_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)  # ✅ Process each frame
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections  # ✅ Return player detections for all frames

    def draw_bboxes(self, video_frames, player_detections, court_keypoints):
        """Draw bounding boxes only for players whose feet are inside the court."""
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                foot_position = ((x1 + x2) // 2, y2)  # Bottom-center of bbox

                if self.is_inside_court(foot_position, court_keypoints):
                    # ✅ Draw bounding box for valid players
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"Player {track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames
