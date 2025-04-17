import cv2
import pandas as pd
import pickle
import os

class shuttlecockTracker:
    def __init__(self, csv_path, valid_frame_indices):
        """Initialize tracker using a CSV file, keeping only valid frames."""
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        # ✅ Keep only frames that exist in valid_frame_indices
        self.df = self.df[self.df["Frame"].isin(valid_frame_indices)]

        # ✅ Set frame index for fast lookups
        self.df.set_index("Frame", inplace=True)

        print(f"✅ Loaded shuttlecock data for {len(self.df)} valid frames.")

    def draw_shuttlecock(self, video_frames, valid_frame_indices):
        """Overlay shuttlecock positions from CSV onto video frames."""
        output_video_frames = []
        valid_frame_count = 0

        for frame_idx, original_frame_idx in enumerate(valid_frame_indices):  
            frame = video_frames[frame_idx]  # Ensure correct frame alignment

            if original_frame_idx in self.df.index:
                x, y = int(self.df.loc[original_frame_idx, "X"]), int(self.df.loc[original_frame_idx, "Y"])

                # ✅ Draw a circle at shuttlecock position
                cv2.circle(frame, (x, y), 15, (0, 255, 255), 3)  # Yellow circle, 3px outline
                valid_frame_count += 1

            output_video_frames.append(frame)

        print(f"✅ Shuttlecock drawn on {valid_frame_count} frames.")
        return output_video_frames
    
    def detect_frames(self, frames, valid_frame_indices, read_from_stub=False, stub_path=None):
        """Detect shuttlecock only in valid play frames."""
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print("✅ Loading shuttlecock detections from stub...")
            with open(stub_path, 'rb') as f:
                shuttlecock_detections = pickle.load(f)

            # ✅ Verify stub consistency
            if isinstance(shuttlecock_detections, list) and all(isinstance(item, tuple) for item in shuttlecock_detections):
                return shuttlecock_detections  # ✅ Already formatted correctly

            print("⚠️ Warning: Stub format mismatch, re-processing detections.")

        # ✅ Process valid frames only
        shuttlecock_detections = []
        for idx, original_idx in enumerate(valid_frame_indices):
            frame = frames[idx]  # Ensure correct alignment
            shuttlecock_dict = self.detect_frame(frame)
            shuttlecock_detections.append((original_idx, shuttlecock_dict))  # ✅ Keep original index

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(shuttlecock_detections, f)
            print(f"✅ Shuttlecock detections saved to {stub_path}")

        return shuttlecock_detections

    def detect_frame(self, frame):
        """Mock shuttlecock detection function (Replace with actual model)."""
        return {"x": 100, "y": 200}  # Placeholder detection (use actual logic)
