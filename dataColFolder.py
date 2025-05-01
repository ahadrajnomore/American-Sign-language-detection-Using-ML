import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

input_folder = "Input/y"  # Input folder path
output_folder = "LandmarkDataFromFolder/Y"
offset = 20
imgSize = 300

detector = HandDetector(maxHands=1)

# Get a list of image file names in the input folder
image_files = os.listdir(input_folder)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

counter = 1

for image_file in image_files:
    # Read the input image from the input folder
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w > 0 and h > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Check if the cropped image dimensions are valid
            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
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

                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, f"asl_{counter}.jpg")
                print(f"Saving processed image {image_file} to {output_path}")
                cv2.imwrite(output_path, imgWhite)

                counter += 1

cv2.destroyAllWindows()
