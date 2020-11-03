from __future__ import print_function
import cv2
import numpy as np
import argparse
import pyautogui

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'HW4 window'
isColor = False


def nothing(x):
    pass


cam = cv2.VideoCapture(0)
#cv2.namedWindow(window_name)
#cv2.createTrackbar(trackbar_type, window_name, 3, max_type, nothing)
# # Create Trackbar to choose Threshold value
#cv2.createTrackbar(trackbar_value, window_name, 0, max_value, nothing)
# # Call the function to initialize
#cv2.createTrackbar(trackbar_blur, window_name, 1, 20, nothing)
# # create switch for ON/OFF functionality
# color_switch = 'Color'
# cv2.createTrackbar(color_switch, window_name, 0, 1, nothing)
# cv2.createTrackbar('Contours', window_name, 0, 1, nothing)
spacePressed = False
while True:
    ret, frame = cam.read()

    #### skinmask snippet
    lower_HSV = np.array([0, 65, 0], dtype="uint8")
    upper_HSV = np.array([25, 255, 255], dtype="uint8")

    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)

    lower_YCrCb = np.array((0, 138, 67), dtype="uint8")
    upper_YCrCb = np.array((255, 173, 133), dtype="uint8")

    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)

    skinMask = cv2.add(skinMaskHSV, skinMaskYCrCb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (1, 1), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    # skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)

    """
    cv2.imshow(window_name, skin)
    k = cv2.waitKey(1)  # k is the key pressed
    if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
         # exit
        cv2.destroyAllWindows()
        cam.release()
        break
    
    """
    gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_OTSU ) 
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  

    
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
    contours=sorted(contours,key=cv2.contourArea,reverse=True)       
    new_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) 
    cv2.imshow(window_name, new_img)
    cv2.waitKey(1)
    if len(contours)>1:  
        fingerCount = 0
        largestContour = contours[0]  

        hull = cv2.convexHull(largestContour, returnPoints = False)     
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])  
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                      
                    if angle <= np.pi / 3:  
                        new_img = cv2.circle(new_img,far,5,[0,0,255],-1) 
                        fingerCount += 1
                    new_img = cv2.line(new_img,start,end,[0,255,0],2)  
        print(fingerCount)
        M = cv2.moments(largestContour)  
        offsetX = 10
        scaleX = 0
        offsetY = 10
        scaleY = 0
        cX = offsetX + scaleX *int(M["m10"] / M["m00"])  
        cY = offsetY + scaleY *int(M["m01"] / M["m00"])  
        # pyautogui.moveTo(cX, cY, duration=0.02, tween=pyautogui.easeInOutQuad)  
        """
        if(fingerCount == 0):
            pyautogui.press('s')
        elif (fingerCount == 1):
            pyautogui.press('d')
        elif (fingerCount == 2):
            pyautogui.press('a')
        elif (fingerCount == 3):
            pyautogui.press('w')
        else:
            pyautogui.press('space')
        """

        handRingArea = statsSortedByArea[-3][0:4]  
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)  
  
        if(isIncreased(handRingArea,prevHandRingArea,threshold)):  
            ZoomIn()  
            prevHandRingArea = handRingArea  
        elif(isDecreased(handRingArea,prevHandRingArea,threshold)):  
            ZoomOut()  
            prevHandRingArea = handRingArea  
  
        if(isIncreased(angle,prevAngle,threshold)):  
            RotateLeft()  
            prevAngle = angle  
        elif(isDecreased(angle,prevAngle,threshold)):  
            RotateRight()  
            prevHandRingArea = handRingArea

        

        cv2.imshow(window_name, new_img)
        k = cv2.waitKey(20)
"""
    markers = np.array(markers, dtype=np.uint8)  
    label_hue = np.uint8(179*markers/np.max(markers))  
    blank_ch = 255*np.ones_like(label_hue)  
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0  
    cv2.imshow(window_name, labeled_img)
    k = cv2.waitKey(1)
    
    statsSortedByArea = stats[np.argsort(stats[:, 4])]  
    if (ret>2):  
        try:  
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):  
                cntLength = len(contours[i])  
                if(cntLength>maxCntLength):  
                    cnt = contours[i]  
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);  
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)  
              
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)  
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt) 
            print((x, y), (MA, ma), angle)
            cv2.imshow("ROI "+str(2), subImg)  
            k = cv2.waitKey(1)  
            if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
                # exit
                cv2.destroyAllWindows()
                cam.release()
                break
        except:  
            print("No hand found")  

    """
def isIncreased(handRingArea,prevHandRingArea,threshold)  
    return (handRingArea > prevHandRingArea+Threshold)  
  
def isDecreased(handRingArea,prevHandRingArea,threshold)  
    return (handRingArea < prevHandRingArea-Threshold)  
  
def ZoomIn():  
    pyautogui.hotkey('ctrl', '+')  
  
def ZoomOut():  
    pyautogui.hotkey('ctrl', '-')  
  
def RotateRight():  
    pyautogui.hotkey('ctrl', 'R')  
  
def RotateLeft():  
    pyautogui.hotkey('ctrl', 'R', presses=3)

 
