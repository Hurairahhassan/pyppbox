#################################################################################
# Example of using modules's YAML files
#################################################################################

import cv2
import time

from pyppbox.standalone import (setMainDetector, setMainTracker, setMainReIDer, 
                                detectPeople, trackPeople, reidPeople)
from pyppbox.utils.visualizetools import visualizePeople


# Using a custom YAML/JSON file allows you to set or adjust the parameters 
# of a specific module easily without changing your code.

# This YAML/JSON file contains only the configurations of a supported detector
mydetector = "single_config/mydetector.yaml"
# mydetector = "single_config/mydetector.json"

# This YAML/JSON file contains only the configurations of a supported tracker
# mytracker = "single_config/mytracker.yaml"
mytracker = "single_config/mytracker.json"

# This YAML/JSON file contains only the configurations of a supported reider
# myreider = "single_config/myreider.yaml"
myreider = "single_config/myreider.json"

setMainDetector(detector=mydetector)
setMainTracker(tracker=mytracker)
setMainReIDer(reider=myreider)

tracking_dict = {}
def update_tracking(people, current_time):
    for person in people:
        deepid = person.deepid
        if deepid not in tracking_dict:
            tracking_dict[deepid] = {'start_time': current_time, 'last_seen': current_time}
        else:
            tracking_dict[deepid]['last_seen'] = current_time


# input_video = r"E:\reid\pyppbox\pyppbox\data\datasets\GTA_V_DATASET\videos\demo_detection2(m).mp4"
cap = cv2.VideoCapture('E:/Object_detection/demo_detection.mp4')

start_time = time.time() 
while cap.isOpened():
    hasFrame, frame = cap.read()

    if hasFrame:
        #current_time = time.time() - start_time  # Current time in seconds from the start of the video

        # Detect people without visualizing
        detected_people, _ = detectPeople(frame, img_is_mat=True, visual=False)

        # Track the detected people
        tracked_people = trackPeople(frame, detected_people, img_is_mat=True)

        # Re-identify the tracked people
        reidentified_people, reid_count = reidPeople(
            frame, 
            tracked_people,
            deduplicate= False,
            img_is_mat=True
        )
        # Update tracking with the current time
        update_tracking(reidentified_people, start_time)

        # Visualize people in video frame with reid status `show_reid=reid_count`
        visualized_mat = visualizePeople(
            frame,
            reidentified_people,
            tracking_dict,
            start_time,
            show_skl= (False, False, 1),
            show_ids= (False, True, False),
            show_reid=(0, 0),
            show_repspoint= False
        )
        cv2.imshow("pyppbox: inferencing.py", visualized_mat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()