import cv2
import numpy as np
import PlantDetectHelper as pdh

# Read the dandelion image
def detectColorDandelion(img):
    imgContour = img.copy()

    colorImg = pdh.applyHSV(img, 0, 37, 123, 255, 206, 255)

    imgGray = cv2.cvtColor(colorImg,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    pdh.getContours(imgCanny, imgContour)

    imgBlank = np.zeros_like(img)
    imgStack = pdh.stackImages(4, [imgContour])

    cv2.imshow("Stack", imgStack)

    cv2.waitKey(0)

def main():
    img = cv2.imread("Resources/Dandelion_Field.jfif")
    detectColorDandelion(img)

if __name__ == "__main__": main()
