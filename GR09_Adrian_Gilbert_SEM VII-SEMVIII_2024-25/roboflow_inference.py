import os
import pickle
import csv
from inference_sdk import InferenceHTTPClient

# Initialize Roboflow API client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="hvPnBUxnbQj5nSsCJCV0"
)

MODEL_ID = "badminton-pose-classification/3"

# File paths
POSE_PREDICTIONS_PATH = "pose_predictions.pkl"
CSV_OUTPUT_PATH = "player1_pose_predictions.csv"

# Define hit frame directory for Player 1
player1_folder = "player1_hit_frames"

def classify_images(image_folder):
    """
    Runs pose classification on all images in the folder and returns predictions.
    """
    results = {}

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        try:
            # Perform classification
            prediction = CLIENT.infer(image_path, model_id=MODEL_ID)

            # Extract actual class prediction
            if "predictions" in prediction and len(prediction["predictions"]) > 0:
                pose_class = prediction["predictions"][0]["class"]
            else:
                pose_class = "Unknown"  # If no prediction, default to Unknown

            # Store result
            frame_number = image_name.split("_")[-1].split(".")[0]  # Extract frame number from filename
            results[frame_number] = pose_class
            print(f"✅ Processed Frame {frame_number}: {pose_class}")

        except Exception as e:
            print(f"⚠️ Error processing {image_name}: {e}")

    return results

# Generate pose predictions for Player 1 only
player1_preds = classify_images(player1_folder)

# Save to pickle (only Player 1 data)
with open(POSE_PREDICTIONS_PATH, "wb") as f:
    pickle.dump({"player1": player1_preds}, f)

print("✅ Player 1 pose predictions saved successfully to pose_predictions.pkl")

# Save results to CSV
with open(CSV_OUTPUT_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Frame Number", "Pose Prediction"])  # CSV header
    for frame, pose in sorted(player1_preds.items(), key=lambda x: int(x[0])):  # Sort by frame number
        writer.writerow([frame, pose])

print(f"✅ Player 1 pose predictions saved successfully to {CSV_OUTPUT_PATH}")
