import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import os

image_folder = os.path.expanduser("~/Desktop/Showcase/emotions")
print(f"Looking for images in: {image_folder}")
print(f"Folder exists: {os.path.exists(image_folder)}")

if os.path.exists(image_folder):
    print(f"Files in folder: {os.listdir(image_folder)}")
else:
    print("ERROR: Folder does not exist!")
    exit()

# Load emotion images
emotion_images = {}
emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']

for emotion in emotions:
    img_path = os.path.join(image_folder, f"{emotion}.jpg")
    print(f"Checking: {img_path}")
    if os.path.exists(img_path):
        emotion_images[emotion] = cv2.imread(img_path)
        print(f"  ✓ Loaded successfully")
    else:
        print(f"  ✗ NOT FOUND")

print(f"\nTotal images loaded: {len(emotion_images)}\n")

if not emotion_images:
    print("ERROR: No images loaded!")
    exit()

# Initialize
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

classifier = pipeline("image-classification", 
                    model="trpakov/vit-face-expression")

current_emotion_image = None
current_emotion = "neutral"

print("\nStarting emotion detection... Press 'q' to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, fw, fh) in faces:
        face_roi = frame[y:y+fh, x:x+fw]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        try:
            predictions = classifier(face_pil)
            if predictions:
                emotion = predictions[0]['label'].lower()
                confidence = predictions[0]['score']
                
                # Update displayed image if emotion changed
                if emotion in emotion_images:
                    current_emotion = emotion
                    current_emotion_image = emotion_images[emotion]
                
                # Draw emotion label on face
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 255), 2)
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Display webcam + emotion image side-by-side
    if current_emotion_image is not None:
        emotion_img_resized = cv2.resize(current_emotion_image, (w, h))
        combined = np.hstack([frame, emotion_img_resized])
        
        # Add emotion label at top
        cv2.putText(combined, f"Current Emotion: {current_emotion.upper()}", 
                (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        combined = frame
    
    cv2.imshow("Your Face | Emotion Image", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()