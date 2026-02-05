import cv2
import dlib
from scipy.spatial import distance

# =========================================================
# LOAD MODELS
# =========================================================

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

# =========================================================
# EYE LANDMARK INDEXES
# =========================================================

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]

# =========================================================
# EAR CALCULATION
# =========================================================

def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================

def detect_drowsiness():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:

            landmarks = predictor(gray, face)

            left_eye = []
            right_eye = []

            # Left eye points
            for i in LEFT_EYE:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                left_eye.append((x, y))

            # Right eye points
            for i in RIGHT_EYE:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                right_eye.append((x, y))

            # Compute EAR
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2

            # Drowsiness condition
            if ear < 0.25:
                cv2.putText(
                    frame,
                    "DROWSY!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
