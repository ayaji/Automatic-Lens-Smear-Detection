import cv2
import numpy as np
import copy
import sys
import glob
import imutils

WIDTH = 500;
HEIGHT = 500

# Provide the input arguments
if len(sys.argv) < 2:
    print('Please provide the absolute path to Image Dataset:\n')
    exit(0)
else:
    imgFiles = glob.glob(sys.argv[1] + "/*.jpg")
    print(str(len(imgFiles)) + ' # files to process')

    # Create a container to save the average image
    avgImg = np.zeros((WIDTH, HEIGHT),np.float)

    for img in imgFiles:
        print("Processing image: {}".format(img))

        # Read the image and resize the image into fixed width and height
        img = cv2.imread(img)
        img = imutils.resize(img, width=WIDTH)

        # Convert the image to grayscale
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Apply histogram equalisation for contrast enhancement
        img = cv2.equalizeHist(img)

        avgImg = avgImg + img

    # Take the average of the image
    avgImg = avgImg / len(imgFiles)
    avgImg = np.array(np.round(avgImg), dtype=np.uint8)

cv2.imshow("Average Image",avgImg)
cv2.imwrite('AverageImage.jpg', avgImg)
cv2.waitKey(0)

# Convert the grayscale image to binary image using adaptive thresholding
thresholdImg = cv2.adaptiveThreshold(avgImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4)
cv2.imshow("Threshold Image",thresholdImg)
cv2.imwrite('ThresholdedImage.jpg', thresholdImg)
cv2.waitKey(0)

# Median Blur to remove the salt - pepper noise
thresholdImg = cv2.medianBlur(thresholdImg, 29)
intImg = thresholdImg.astype("uint8") * 255

# Detecting the edges in the image
edgeMapImage = cv2.Canny(intImg, 9, 50, apertureSize=5, L2gradient=True)

# Detecting the Contours
_, contours, _ = cv2.findContours(edgeMapImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a list of contours found
contourList = []
for i in contours:
    contourList.append(i)

# Create a placeholder for masked image
maskImg = np.zeros((WIDTH, HEIGHT, 1), np.float)

# Copy the averaged image to the smear image
smearImg = copy.deepcopy(avgImg)

# Read a test image from the dataset to identify the smear
testImg = cv2.imread(imgFiles[217])
testImg = imutils.resize(testImg, width=WIDTH)

testImgOrg = cv2.imread(imgFiles[217])
testImgOrg = imutils.resize(testImgOrg, width=WIDTH)

if len(contourList) > 0:
    # Draw the contours on the mask image, smear image and test image
    cv2.drawContours(maskImg, contours, -1, (255, 55, 255), 15)
    cv2.drawContours(smearImg, contours, -1, (255, 55, 0), 3)
    cv2.drawContours(testImg, contours, -1, (255, 55, 0), 3)

    # Show amd write the masked image
    cv2.imshow("Masked Image", maskImg)
    cv2.imwrite('MaskedImage.jpg',maskImg)
    cv2.waitKey(0)

    # Show and write the smear image
    cv2.imshow("Smear Image", smearImg)
    cv2.imwrite('SmearImage.jpg', smearImg)
    cv2.waitKey(0)

    # Show amd write the detected test image
    result = np.vstack([testImgOrg, testImg])
    cv2.imshow("Smear Detection",result)
    cv2.imwrite('FinalImage.jpg', testImg)
    cv2.waitKey(0)

    print('Smear Detected. Result in FinalImage.jpg')
else:
    print('No Smear Detected')