import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------
# CONFIGURATION
# ---------------------------

MODEL_PATH="face_landmarker.task"
ALPHA=0.4
LEFT_THRESHOLD=0.40
RIGHT_THRESHOLD=0.46


# ---------------------------
# INITIALIZE MEDIAPIPE
# ---------------------------

def initialize_landmarker():
    BaseOptions=mp.tasks.BaseOptions
    FaceLandmarker=vision.FaceLandmarker
    FaceLandmarkerOptions=vision.FaceLandmarkerOptions
    RunningMode=vision.RunningMode

    options=FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),running_mode=RunningMode.VIDEO,num_faces=1)

    return FaceLandmarker.create_from_options(options)


# ---------------------------
# GAZE CLASSIFICATION
# ---------------------------

def classify_gaze(position):
    if position<LEFT_THRESHOLD:
        return "Looking Left"
    elif position>RIGHT_THRESHOLD:
        return "Looking Right"
    else:
        return "Center"


# ---------------------------
# PROCESS FRAME
# ---------------------------

def process_frame(frame, landmarker,smoothed_position, timestamp):

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    result=landmarker.detect_for_video(mp_image,timestamp)

    gaze=None

    if result.face_landmarks:
        landmarks=result.face_landmarks[0]

        # Right eye landmarks
        outer_corner=landmarks[33]
        inner_corner=landmarks[133]
        iris=landmarks[468]

        eye_width=inner_corner.x-outer_corner.x

        if eye_width!=0:
            normalized_position=((iris.x-outer_corner.x)/eye_width)

            # Exponential smoothing
            smoothed_position=(ALPHA*normalized_position+(1-ALPHA)*smoothed_position)
            gaze=classify_gaze(smoothed_position)

            # Draw iris
            h, w, _=frame.shape
            iris_px=int(iris.x * w)
            iris_py=int(iris.y * h)
            cv2.circle(frame,(iris_px, iris_py),4,(0,255,0),-1)

    return frame,gaze,smoothed_position


# ---------------------------
# MAIN LOOP
# ---------------------------

def main():

    cap=cv2.VideoCapture(0)
    landmarker=initialize_landmarker()

    smoothed_position=0.5
    timestamp=0

    while True:
        isTrue,frame=cap.read()
        if not isTrue:
            break

        # Mirror frame for natural interaction
        frame=cv2.flip(frame,1)

        frame,gaze,smoothed_position=process_frame(frame,landmarker,smoothed_position,timestamp)

        timestamp+=1

        if gaze:
            cv2.putText(frame,gaze,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("MediaPipe Gaze Tracker",frame)

        if cv2.waitKey(1)&0xFF==ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# ENTRY POINT
# ---------------------------

if __name__=="__main__":
    main()