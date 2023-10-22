import cv2
import component
import numpy as np

def process_image_and_get_indices(image_path):
    #parameter
    # pathImage = "/Users/qadirrun/Documents/ETL/img/custom_declaration_2.png"  
    pathImage = image_path
    heightImg = 700
    widthImg = 700
    question = 9
    choice = 2

    img = cv2.imread(pathImage)

    #preprocess
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)
    imgcont = img.copy()

    #focus
    secW = 530  # Width of each section
    secH = 380  # Height of each section

    # Coordinates for the "d" section in the bottom right quadrant
    x1, y1 = secW, secH  # Starting coordinates
    x2, y2 = widthImg, heightImg  # Ending coordinates

    # Crop the "d" section
    cropped_img = img[y1:y2, x1:x2]
    cropped_imges = cv2.resize(cropped_img, (widthImg, heightImg))
    # cropped_img_gray = cv2.cvtColor(cropped_imges, cv2.COLOR_BGR2GRAY)
    imgWarpGray = cv2.cvtColor(cropped_imges, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    #contour
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgcont, contours, -1,(0,255,0),1)

    # component.rectContour(contours)
    oval_contours=component.ovalContour(contours, max_deviation=1)
    # print(f"Number of oval/circular contours found: {len(oval_contours)}")

    boxes = component.splitBoxes(imgThresh)

    #get pixel value yes=0, no=1
    myPixelVal = np.zeros((question,choice))
    countC = 0
    countR = 0
    
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]=totalPixels
        countC +=1
        if (countC== choice): countR +=1 ;countC=0
    # print(myPixelVal)

    myIndex = []
    for x in range (0,question):
        arr = myPixelVal[x]
        # print(arr)
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    print(myIndex)
    return myIndex


    # imgblank = np.zeros_like(img)
    # cv2.drawContours(img, oval_contours, -1, (0, 255, 0), thickness=cv2.FILLED)
    # imageArray =([img,imgGray,imgBlur,imgCanny],[imgcont,imgblank,imgblank,cropped_imges])
    # imgstack = component.stackImages(imageArray, 0.5)

    # cv2.imshow("ori",imgstack)
    # cv2.waitKey(0)
    
