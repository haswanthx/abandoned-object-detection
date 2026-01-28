import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
import os
import time

# =============================================================================
# 1. Advanced Centroid Tracker
# =============================================================================
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)


            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# =============================================================================
# 2. Main Execution Logic
# =============================================================================

# CONFIGURATION
VIDEO_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.6      # Increased to 0.6 for better accuracy
DISTANCE_THRESHOLD = 150        # Distance to consider "Owner"
TIME_THRESHOLD = 60             # Frames before alarm (approx 3 seconds)
MOVEMENT_THRESHOLD = 10         # Pixels. If bag moves more than this, reset timer.

# LOAD MODEL
print("[INFO] Loading YOLO model...")
model = YOLO('yolov8n.pt') 

# YOLO Classes: 0: person, 24: backpack, 26: handbag, 28: suitcase
TARGET_CLASSES = [0, 24, 26, 28]

# Trackers
person_tracker = CentroidTracker(maxDisappeared=20)
bag_tracker = CentroidTracker(maxDisappeared=50)

# STATE VARIABLES
bag_timers = {}           # Stores how long a bag has been alone
bag_start_locations = {}  # Stores where the bag was when it first became "alone"
alert_sent = {}           # To prevent spamming alerts

cap = cv2.VideoCapture(VIDEO_SOURCE)

def trigger_alert(frame, bag_id):
    """
    1. Plays a Mac System Sound
    2. Saves an Evidence Photo
    """
    if bag_id not in alert_sent:
        print(f"[ALERT] Abandoned Object Detected! ID: {bag_id}")
        
        # 1. PLAY SOUND (Mac Specific Command)
        # 'Glass' is a standard Mac sound. You can use 'Ping', 'Basso', etc.
        os.system('afplay /System/Library/Sounds/Glass.aiff &')
        
        # 2. SAVE EVIDENCE IMAGE
        filename = f"alert_bag_{bag_id}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Evidence saved: {filename}")
        
        alert_sent[bag_id] = True

while True:
    ret, frame = cap.read()
    if not ret: break

    # Resize for speed
    frame = cv2.resize(frame, (1000, 600))
    
    # Run YOLO Detection
    results = model(frame, verbose=False)[0]
    
    rects_persons = []
    rects_bags = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if conf > CONFIDENCE_THRESHOLD:
            if cls == 0: 
                rects_persons.append((x1, y1, x2, y2))
            elif cls in [24, 26, 28]: 
                rects_bags.append((x1, y1, x2, y2))

    objects_persons = person_tracker.update(rects_persons)
    objects_bags = bag_tracker.update(rects_bags)

    # --- ADVANCED LOGIC ---
    for (bagID, bagCentroid) in objects_bags.items():
        bag_owner = None
        min_dist = 99999

        # 1. Find Owner
        for (personID, personCentroid) in objects_persons.items():
            d = dist.euclidean(bagCentroid, personCentroid)
            if d < min_dist:
                min_dist = d
                if d < DISTANCE_THRESHOLD:
                    bag_owner = personID
        
        # 2. Logic Decision
        if bag_owner is not None:
            # Bag has owner -> RESET EVERYTHING
            bag_timers[bagID] = 0
            if bagID in bag_start_locations: del bag_start_locations[bagID]
            if bagID in alert_sent: del alert_sent[bagID]
            
            # Visuals
            color = (0, 255, 0) # Green
            cv2.line(frame, tuple(bagCentroid), tuple(objects_persons[bag_owner]), color, 2)
            cv2.putText(frame, f"Owner: {bag_owner}", (bagCentroid[0], bagCentroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        else:
            # Bag is Ownerless... BUT is it stationary?
            if bagID not in bag_start_locations:
                bag_start_locations[bagID] = bagCentroid # Record initial position
            
            # Check how far it has moved since it became ownerless
            start_pos = bag_start_locations[bagID]
            movement = dist.euclidean(bagCentroid, start_pos)
            
            if movement > MOVEMENT_THRESHOLD:
                # Bag is moving (likely being carried or dragged), so RESET timer
                bag_timers[bagID] = 0
                bag_start_locations[bagID] = bagCentroid
                status = "Moving..."
                color = (255, 255, 0) # Cyan
            else:
                # Bag is NOT moving AND Ownerless -> DANGER
                bag_timers[bagID] = bag_timers.get(bagID, 0) + 1
                status = f"Timer: {bag_timers[bagID]}"
                color = (0, 165, 255) # Orange

                # 3. TRIGGER ALARM
                if bag_timers[bagID] > TIME_THRESHOLD:
                    trigger_alert(frame, bagID)
                    status = "ABANDONED!"
                    color = (0, 0, 255) # Red
                    cv2.putText(frame, "ALARM TRIGGERED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            cv2.circle(frame, tuple(bagCentroid), 5, color, -1)
            cv2.putText(frame, status, (bagCentroid[0], bagCentroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Abandoned Object - Advanced", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()