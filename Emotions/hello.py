import cv2
from transformers import pipeline
from PIL import Image

cap = cv2.VideoCapture(0)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load emotion classifier
classifier = pipeline("image-classification", 
                    model="trpakov/vit-face-expression")

emotions_colors = {
    'angry': (0, 0, 255),
    'happy': (0, 255, 255),
    'neutral': (128, 128, 128),
    'sad': (255, 0, 255),
    'surprise': (255, 255, 0),
    'fear': (255, 0, 0),
    'disgust': (0, 255, 0)
}

print("Starting emotion detection... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        try:
            # Classify emotion
            predictions = classifier(face_pil)
            if predictions:
                emotion = predictions[0]['label'].lower()
                confidence = predictions[0]['score']
                color = emotions_colors.get(emotion, (255, 255, 255))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Confidence bar
                bar_width = int(confidence * w)
                cv2.rectangle(frame, (x, y+h+5), 
                            (x+bar_width, y+h+15), color, -1)
        except Exception as e:
            print(f"Error classifying: {e}")
    
    cv2.imshow("Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
