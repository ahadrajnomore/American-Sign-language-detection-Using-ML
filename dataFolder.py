import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

image_path = "A/3001.jpg"  # Path to your input image
output_folder = "P"  # Output folder path
offset = 20
imgSize = 300

detector = HandDetector(maxHands=1)

# Read the input image
img = cv2.imread(image_path)

hands, img = detector.findHands(img)
if hands:
    hand = hands[0]
    x, y, w, h = hand['bbox']

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    imgCropShape = imgCrop.shape

    aspectRatio = h / w

    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        imgResizeShape = imgResize.shape
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wCal + wGap] = imgResize

    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        imgResizeShape = imgResize.shape
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hCal + hGap, :] = imgResize

    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "Processed_Image.jpg")
    print(f"Saving processed image to {output_path}")

    # Save the processed image to the output folder
    cv2.imwrite(output_path, imgWhite)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
