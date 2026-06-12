import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from tensorflow.keras.models import load_model

# ==================================================
# Load Models and Class Labels
# ==================================================
def load_model_and_classes(model_path, classes_file):
    model = load_model(model_path)
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return model, classes

dress_model, dress_classes = load_model_and_classes("dress_model.h5", "dress_classes.txt")
grooming_model, grooming_classes = load_model_and_classes("grooming_model.h5", "grooming_classes.txt")
posture_model, posture_classes = load_model_and_classes("posture_model.h5", "posture_classes.txt")
id_model, id_classes = load_model_and_classes("id_model.h5", "id_classes.txt")

# ==================================================
# Advanced Consistency Tracker
# ==================================================
class AdvancedConsistencyTracker:
    def __init__(self, consistency_threshold=10, memory_duration=30):
        self.current_predictions = None
        self.consistent_frames = 0
        self.consistency_threshold = consistency_threshold
        self.memory_duration = memory_duration
        self.current_person_id = "person_1"
        self.last_detection_time = time.time()
        self.prediction_history = []
        
    def calculate_similarity(self, pred1, pred2):
        """Calculate similarity between two prediction sets"""
        if pred1 is None or pred2 is None:
            return 0
            
        matches = 0
        total = 0
        for key in pred1:
            if key in pred2:
                if pred1[key] == pred2[key]:
                    matches += 1
                total += 1
        return matches / total if total > 0 else 0
    
    def detect_person_continuity(self, new_predictions):
        """Detect if it's the same person based on prediction patterns"""
        current_time = time.time()
        
        # Clean old history
        self.prediction_history = [p for p in self.prediction_history 
                                 if current_time - p['time'] < self.memory_duration]
        
        if not self.current_predictions:
            # First detection
            self.current_predictions = new_predictions
            self.consistent_frames = 1
            self.last_detection_time = current_time
            self.prediction_history.append({
                'time': current_time,
                'predictions': new_predictions
            })
            return True, "First Detection"
        
        similarity = self.calculate_similarity(self.current_predictions, new_predictions)
        
        if similarity >= 0.6:  # Same person (most predictions match)
            self.consistent_frames += 1
            self.last_detection_time = current_time
            
            # After threshold, return cached predictions for consistency
            if self.consistent_frames >= self.consistency_threshold:
                return True, f"Consistent (Similarity: {similarity:.2f})"
            else:
                self.current_predictions = new_predictions
                return True, f"Stabilizing ({self.consistent_frames}/{self.consistency_threshold})"
        else:
            # Significant change - likely new person
            self.current_predictions = new_predictions
            self.consistent_frames = 1
            self.last_detection_time = current_time
            self.current_person_id = f"person_{int(time.time())}"  # Unique ID based on timestamp
            return False, f"New Person Detected (Similarity: {similarity:.2f})"
    
    def get_predictions(self, new_predictions):
        is_same_person, status = self.detect_person_continuity(new_predictions)
        
        # If same person and we're beyond consistency threshold, return cached predictions
        if is_same_person and self.consistent_frames >= self.consistency_threshold:
            return self.current_predictions, status
        else:
            return new_predictions, status
    
    def get_person_id(self):
        return self.current_person_id

# ==================================================
# Prediction Function
# ==================================================
def predict_all(frame):
    img = cv2.resize(frame, (128, 128))
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = {}
    preds["dresscode"] = dress_classes[np.argmax(dress_model.predict(img_array))]
    preds["bear"] = grooming_classes[np.argmax(grooming_model.predict(img_array))]
    preds["posture"] = posture_classes[np.argmax(posture_model.predict(img_array))]
    preds["card"] = id_classes[np.argmax(id_model.predict(img_array))]
    return preds

# ==================================================
# Setup Directories and CSV Log
# ==================================================
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, "log.csv")

if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp", "person_id", "image_path", "dresscode", "bear", "posture", "card", "status"]
        writer.writerow(header)

# ==================================================
# Initialize Tracker
# ==================================================
tracker = AdvancedConsistencyTracker(consistency_threshold=8, memory_duration=30)

# ==================================================
# Real-Time Webcam Detection
# ==================================================
cap = cv2.VideoCapture(0)
save_interval = 5  # seconds
last_save_time = time.time()
frame_count = 0

print("[INFO] Starting Advanced Consistency Detection...")
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get new predictions from model
    new_preds = predict_all(frame)
    
    # Get consistent predictions based on person tracking
    current_preds, status = tracker.get_predictions(new_preds)
    person_id = tracker.get_person_id()

    # Display information on frame
    y_offset = 30
    line_height = 25
    
    # Person ID and status
    cv2.putText(frame, f"Person: {person_id}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, y_offset + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    
    # Predictions
    for i, (category, value) in enumerate(current_preds.items()):
        y_pos = y_offset + (i + 2) * line_height
        cv2.putText(frame, f"{category}: {value}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame count and consistency info
    frame_count += 1
    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("AI Discipline Checking System", frame)

    # Auto-save every interval
    current_time = time.time()
    if current_time - last_save_time >= save_interval:
        # Create unique filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"detection_{person_id}_{timestamp_str}.jpg"
        img_path = os.path.join(save_dir, img_name)

        # Save image
        cv2.imwrite(img_path, frame)

        # Log to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp, 
            person_id, 
            img_path, 
            current_preds["dresscode"], 
            current_preds["bear"], 
            current_preds["posture"], 
            current_preds["card"],
            status
        ]

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[SAVED] {timestamp} | Person: {person_id} | Status: {status}")
        last_save_time = current_time

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n[INFO] Detection stopped successfully!")
print(f"[INFO] Images saved in: {save_dir}")
print(f"[INFO] Log file: {log_path}")