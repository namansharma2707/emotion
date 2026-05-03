import cv2
# Dhyan rakhna: Ye 'get_emotion' hum emotion_engine.py me banayenge next step me
from emotion_engine import get_emotion 

def main():
    # Webcam start kar rahe hain (0 matlab default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera open nahi ho pa raha hai.")
        return

    print("Camera start ho gaya hai. Band karne ke liye 'q' dabayein.")

    # Face detect karne ke liye OpenCV ka default Haar Cascade load kar rahe hain
    # (Ye offline aur fast face tracking ke liye best hai)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Camera se ek frame (image) read karo
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame read nahi ho pa raha.")
            break

        # Frame ko grayscale me convert karo (face detection fast karne ke liye)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Chehre detect karo (ek sath multiple faces bhi detect kar lega)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Chehre ke charo taraf ek rectangle banao (Blue color)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Chehre wale hisse ko crop karo taaki DeepFace ko sirf chehra bheja jaye
            face_crop = frame[y:y+h, x:x+w]

            # Apne engine se mood pata karo
            mood = get_emotion(face_crop)

            # Mood ka text rectangle ke upar likho (Green color)
            cv2.putText(frame, mood, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Final video feed screen par dikhao
        cv2.imshow('Offline Mood Detector', frame)

        # Agar keyboard pe 'q' dabaya, toh loop break kar do aur band kar do
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Camera hardware ko free kar do aur windows close kar do
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
