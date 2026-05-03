from deepface import DeepFace

def get_emotion(face_crop):
    
    try:
        
        result = DeepFace.analyze(
            img_path = face_crop, 
            actions = ['emotion'], 
            enforce_detection = False 
        )
        
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion.capitalize()
        
    except Exception as e:
        return "Analyzing..."
