import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/American_sign_language.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Create a variable to store the predicted letters
text = ""
show_instructions = True

window_name = "Sign language detection"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w <= 0 or h <= 0:
            print("Invalid hand region.")
            cv2.putText(imgOutput, "Invalid hand region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(window_name, imgOutput)
            continue

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size == 0:
            print("Empty hand region.")
            cv2.putText(imgOutput, "Empty hand region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(window_name, imgOutput)
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = labels[index]

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        if cv2.waitKeyEx(1) & 0xFF == ord('c') or cv2.waitKeyEx(1) & 0xFF == ord('C'):
            text += label

    cv2.putText(imgOutput, "Text: " + text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if show_instructions:
        # Instructions
        cv2.putText(imgOutput, "Instructions:", (10, imgOutput.shape[0] - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imgOutput, "Press C/c to add", (10, imgOutput.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imgOutput, "Press D/d or Backspace to remove", (10, imgOutput.shape[0] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imgOutput, "Press R/r to remove all", (10, imgOutput.shape[0] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imgOutput, "Press Space to add space", (10, imgOutput.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(imgOutput, "Press S/s to Speak", (10, imgOutput.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow(window_name, imgOutput)

    key = cv2.waitKeyEx(1)
    if key == ord('r') or key == ord('R'):
        text = ""
    elif key == ord('s') or key == ord('S'):
        engine.say(text)
        engine.runAndWait()
    elif key == ord('d') or key == ord('D') or key == 8:  # 8 is the ASCII value of Backspace
        if len(text) > 0:
            text = text[:-1]
    elif key == ord(' '):
        text += " "
        print("Space is pressed")
    elif key == ord('i') or key == ord('I'):
        show_instructions = not show_instructions
        imgOutput = img.copy()

    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
