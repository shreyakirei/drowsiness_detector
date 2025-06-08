import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Drowsiness Detection using MediaPipe")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR calculation helper
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    coords = [(int(landmarks.landmark[i].x * image_w), int(landmarks.landmark[i].y * image_h)) for i in eye_indices]

    # Calculate distances
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # vertical distances
    A = euclidean(coords[1], coords[5])
    B = euclidean(coords[2], coords[4])
    # horizontal distance
    C = euclidean(coords[0], coords[3])

    ear = (A + B) / (2.0 * C)
    return ear

# Eye indices for landmarks from MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

score = 0
threshold = 0.25  # EAR threshold for closed eye
frame_threshold = 15  # Number of consecutive frames eyes must be closed for alert

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        image_h, image_w = frame.shape[:2]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, image_w, image_h)
                right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, image_w, image_h)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < threshold:
                    score += 1
                else:
                    score = max(score - 1, 0)

                # Draw eye contours
                for idx in LEFT_EYE + RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * image_w)
                    y = int(face_landmarks.landmark[idx].y * image_h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Show EAR
                cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Drowsiness alert
                if score > frame_threshold:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            score = max(score - 1, 0)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.write("Check the box to start webcam and detect drowsiness.")

