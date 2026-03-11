import cv2
import mediapipe as mp
import numpy as np




# MediaPipe 
model_path = r"pose_landmarker_full.task"


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)


def calculate_angle(a, b, c):


    a = np.array(a)
    b = np.array(b)
    c = np.array(c)


    ba = a - b
    bc = c - b


    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)


    angle = np.arccos(cosine)


    return np.degrees(angle)





def extract_angles(video_path):


    cap = cv2.VideoCapture(video_path)


    angles = []


    frame_timestamp = 0
    
    with PoseLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            result = landmarker.detect_for_video(
                mp_image,
                frame_timestamp
            )
            frame_timestamp += int(1000 / 30)
            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]
                shoulder = pose_landmarks[11]
                elbow = pose_landmarks[13]
                wrist = pose_landmarks[15]
                angle = calculate_angle(
                    [shoulder.x, shoulder.y],
                    [elbow.x, elbow.y],
                    [wrist.x, wrist.y]
                )
                if not np.isnan(angle):
                    angles.append(round(angle,2))


    cap.release()
    return angles



def compare_angles(reference_angles, patient_angles):
    
    if len(reference_angles) == 0 or len(patient_angles) == 0:
        return 0, 0, "No pose detected"

    min_length = min(len(reference_angles), len(patient_angles))

    differences = []

    for i in range(min_length):
        diff = abs(reference_angles[i] - patient_angles[i])
        differences.append(diff)

    avg_difference = np.mean(differences)

    accuracy_score = max(0, 100 - avg_difference)

    mistake_count = sum(1 for d in differences if d > 20)

    if accuracy_score >= 90:
        feedback = "Excellent movement"
    elif accuracy_score >= 75:
        feedback = "Good but needs improvement"
    else:
        feedback = "Movement range is limited"

    return round(accuracy_score,2), mistake_count, feedback