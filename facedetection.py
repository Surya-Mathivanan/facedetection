# Import OpenCV and other necessary libraries
import cv2

# Dummy gender classification function
def classify_gender(face_image):
    # Replace this with your actual model prediction logic
    # For demonstration, we return "Men" or "Women" randomly
    return "Men" if face_image.shape[1] % 2 == 0 else "Men"

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Real-time face detection and gender classification
while True:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]  # Extract the face region
            gender = classify_gender(face_image)  # Classify gender (replace with your model)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Face Detection with Gender', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Unable to read frame")
        break

cap.release()
cv2.destroyAllWindows()
