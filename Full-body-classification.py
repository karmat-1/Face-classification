import cv2
import mediapipe as mp
from deepface import DeepFace

# MediaPipe init
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe detections
        face_results = face_detection.process(img_rgb)
        pose_results = pose.process(img_rgb)
        hands_results = hands.process(img_rgb)

        # Face detection + deep analysis
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

                # crop face safely
                face_crop = frame[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
                if face_crop.size > 0:
                    try:
                        # DeepFace full analysis
                        result = DeepFace.analyze(
                            face_crop,
                            actions=['age', 'gender', 'race', 'emotion'],
                            enforce_detection=False
                        )
                        info = result[0]

                        # Collect attributes
                        age = info.get("age", "N/A")
                        gender = info.get("dominant_gender", "N/A")
                        race = info.get("dominant_race", "N/A")
                        emotion = info.get("dominant_emotion", "N/A")

                        # Celebrity look-alike (requires DeepFace.find with db)
                        # Here we just simulate, unless you build a celeb DB
                        celeb_label = "Celebrity-LookAlike: N/A"

                        # Display beside bounding box
                        text_lines = [
                            f"Age: {age}",
                            f"Gender: {gender}",
                            f"Race: {race}",
                            f"Emotion: {emotion}",
                            celeb_label
                        ]

                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

                        # Print attributes stacked vertically
                        for i, line in enumerate(text_lines):
                            cv2.putText(frame, line, (x, y - 10 - i*20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 255, 0), 2)

                    except Exception as e:
                        print("DeepFace error:", e)

        # Pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Hands landmarks
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show output
        cv2.imshow("(Full Analysis)", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
