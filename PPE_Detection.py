from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1) # For webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("ppe-1-1.mp4")  # For video file


model = YOLO(".\train_results\weights\best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

myColor = (0, 0, 255)  # Color for bounding box and text
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # Draw rectangle around the object
            w, h = x2 - x1, y2 - y1  # Width and height of the bounding box
            # cvzone.cornerRect(img, (x1, y1, w, h))  # Draw rounded rectangle

            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)


                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', 
                                    (max(0, x1), max(35, y1)), scale=0.5, thickness=1, colorB=myColor,
                                    colorT=(255, 255, 255), colorR=myColor, offset=5)  # Display class name and confidence score
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)  # Draw rectangle around the object
    
    cv2.imshow("Image", img)  # Show the image with bounding boxes and labels
    cv2.waitKey(1)  # Wait for a key press to continue