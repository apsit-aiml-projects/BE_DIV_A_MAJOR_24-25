import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import json
import os
from collections import Counter
from tqdm import tqdm

class PosePredictor:
    def __init__(self, video_path="input_videos/input_video.mp4", csv_path="player1_pose_predictions.csv",
                 output_video_path="output_video.mp4", cache_file="pose_cache.json", combined_image="combined_chart.png"):
        self.video_path = video_path
        self.csv_path = csv_path
        self.output_video_path = output_video_path
        self.cache_file = cache_file
        self.combined_image = combined_image
        self.pose_counts = None
        self.dynamic_pose_counts = Counter()
        self.frame_predictions = {}
        self.current_prediction = "Waiting..."
        self.df = None
        self.cap = None
        self.out = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.total_frames = 0
        
    def get_file_hash(self, filename):
        hasher = hashlib.md5()
        with open(filename, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def load_cached_data(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
                if cache["file_hash"] == self.get_file_hash(self.csv_path):
                    return cache["pose_counts"]
        return None

    def save_cache(self):
        cache = {
            "file_hash": self.get_file_hash(self.csv_path),
            "pose_counts": self.pose_counts
        }
        with open(self.cache_file, "w") as f:
            json.dump(cache, f)

    def load_data(self):
        self.df = pd.read_csv(self.csv_path).sort_values(by="Frame Number")
        self.pose_counts = self.load_cached_data()
        if self.pose_counts is None:
            print("🆕 Computing pose frequencies (first time or CSV changed)...")
            self.pose_counts = self.df["Pose Prediction"].value_counts().to_dict()
            self.save_cache()
        else:
            print("⚡ Loaded cached pose data. Skipping computation.")
        self.frame_predictions = dict(zip(self.df["Frame Number"], self.df["Pose Prediction"]))
    
    def setup_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
    
    def generate_combined_graph(self):
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.pie(self.dynamic_pose_counts.values(), labels=self.dynamic_pose_counts.keys(), autopct="%1.1f%%",
                startangle=140, colors=sns.color_palette("pastel"),
                textprops={'fontsize': 12, 'fontweight': 'bold'})
        plt.title("Pose Distribution", fontweight='bold', fontsize=14)
        plt.subplot(2, 1, 2)
        bars = plt.bar(self.dynamic_pose_counts.keys(), self.dynamic_pose_counts.values(), color="skyblue")
        plt.title("Pose Frequency", fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval)}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.combined_image, bbox_inches='tight', dpi=100)
        plt.close()
    
    def overlay_graph_on_frame(self, frame):
        if os.path.exists(self.combined_image):
            combined_chart_img = cv2.imread(self.combined_image)
            original_height, original_width = combined_chart_img.shape[:2]
            scale_factor = 0.75
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            combined_chart_img = cv2.resize(combined_chart_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            x_offset = self.frame_width - new_width - 20
            y_offset = 120
            frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = combined_chart_img
        return frame

    def process_video(self):
        self.load_data()
        self.setup_video()
        for frame_idx in tqdm(range(self.total_frames), desc="Processing Video"):
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx in self.frame_predictions:
                self.current_prediction = self.frame_predictions[frame_idx]
                self.dynamic_pose_counts[self.current_prediction] += 1
                self.generate_combined_graph()
            overlay = frame.copy()
            pose_box_x, pose_box_y, pose_box_width, pose_box_height = self.frame_width - 280 - 20, 20, 280, 80
            cv2.rectangle(overlay, (pose_box_x, pose_box_y),
                          (pose_box_x + pose_box_width, pose_box_y + pose_box_height), (0, 0, 0), -1)
            cv2.putText(overlay, f"Pose: {self.current_prediction}", (pose_box_x + 10, pose_box_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            frame = self.overlay_graph_on_frame(frame)
            self.out.write(frame)
        self.cap.release()
        self.out.release()
        print(f"✅ Video saved as {self.output_video_path}")
