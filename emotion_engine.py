from deepface import DeepFace

def get_emotion(face_crop):
    """
    Ye function main.py se cropped face image lega, 
    DeepFace model se offline process karega, aur dominant mood return karega.
    """
    try:
        # DeepFace.analyze me hum sirf 'emotion' detect karne ko keh rahe hain taaki processing fast ho.
        # enforce_detection=False isliye rakha hai kyunki main.py ne pehle hi OpenCV se face dhundh kar crop kar diya hai.
        result = DeepFace.analyze(
            img_path = face_crop, 
            actions = ['emotion'], 
            enforce_detection = False 
        )
        
        # DeepFace ek list return karta hai (agar multiple log ho toh).
        # Hum pehle object [0] me se 'dominant_emotion' nikal rahe hain.
        dominant_emotion = result[0]['dominant_emotion']
        
        # Output ko thoda clean dikhane ke liye Capitalize kar rahe hain (e.g., 'happy' -> 'Happy')
        return dominant_emotion.capitalize()
        
    except Exception as e:
        # Agar processing me koi issue aaye (jaise frame bohot blur ho), toh system crash na ho
        # isliye hum default "Analyzing..." ya "Unknown" bhej denge.
        return "Analyzing..."
