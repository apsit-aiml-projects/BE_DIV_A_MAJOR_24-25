import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np
from scipy.interpolate import interp1d
import pickle
import os

class CourtLineDetector:
    def __init__(self, model_path, pickle_path="tracker_stubs/court_keypoints.pkl"):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 8)  
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.pickle_path = pickle_path  # Path to save/load keypoints
        self.keypoints = None  # Stores the loaded or manually selected keypoints

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """ Predict court keypoints using the model. """
        if self.load_keypoints():  # Load from pickle if available
            return self.keypoints  

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()

        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        self.keypoints = keypoints
        self.save_keypoints()  # Save predicted keypoints to pickle
        return keypoints

    def save_keypoints(self):
        """ Save manually selected or predicted keypoints to a pickle file. """
        with open(self.pickle_path, "wb") as f:
            pickle.dump(self.keypoints, f)
        print(f"Saved court keypoints to {self.pickle_path}")

    def load_keypoints(self):
        """ Load court keypoints from pickle if available. """
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                self.keypoints = pickle.load(f)
            print(f"Loaded court keypoints from {self.pickle_path}")
            return True
        return False

    def manually_select_keypoints(self, frame):
        """ Allow the user to manually click keypoints on the frame. """
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Court Points", frame)

                if len(points) == 4:  # Assuming 4 keypoints needed
                    cv2.destroyAllWindows()

        print("Click on the four court corners in order (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")
        cv2.imshow("Select Court Points", frame)
        cv2.setMouseCallback("Select Court Points", click_event)
        cv2.waitKey(0)

        if len(points) == 4:
            self.keypoints = np.array([p for point in points for p in point])  # Flatten list
            self.save_keypoints()  # Save manually selected points
            return self.keypoints
        else:
            print("Error: Not enough points selected!")
            return None

    def draw_keypoints(self, image, keypoints):
        """ Draw keypoints and a bounding box connecting them. """
        
        if keypoints is None or len(keypoints) < 8:
            print("⚠️ Not enough keypoints to draw a court bounding box!")
            return image

        # Convert flattened keypoints into (x, y) coordinate pairs
        points = [(int(keypoints[i]), int(keypoints[i+1])) for i in range(0, len(keypoints), 2)]

        # ✅ Draw circles for keypoints
        for idx, (x, y) in enumerate(points):
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red dots
            cv2.putText(image, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Connect keypoints to form a **bounding box**
        if len(points) == 4:
            cv2.line(image, points[0], points[1], (255, 0, 0), 2)  # Top side (Blue)
            cv2.line(image, points[1], points[3], (255, 0, 0), 2)  # Right side
            cv2.line(image, points[3], points[2], (255, 0, 0), 2)  # Bottom side
            cv2.line(image, points[2], points[0], (255, 0, 0), 2)  # Left side

        return image


    def draw_keypoints_on_video(self, video_frames, keypoints):
        """ Draw keypoints + bounding box on every frame in the video. """
        
        output_video_frames = []
        for frame in video_frames:
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame_with_keypoints)
        
        return output_video_frames

