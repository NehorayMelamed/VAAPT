### https://github.com/nicknochnack/MediaPipePoseEstimation/blob/main/Media%20Pipe%20Pose%20Tutorial.ipynb

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




import cv2
import mediapipe as mp
import numpy as np

# Initiate mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load image
image = cv2.imread("/Grounded_Segment_Anything/outputs7/grounded_sam_output.jpg")  # replace with your image path

# Initiate pose estimation model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Convert image back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        print('Left shoulder:')
        print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
        print('Right shoulder:')
        print(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    except:
        print("No pose landmarks detected.")

    # Render detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    # Display image
    image = cv2.resize(image, (200,400))
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image



