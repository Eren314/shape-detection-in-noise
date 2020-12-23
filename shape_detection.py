import cv2
import numpy as np
from matplotlib import pyplot as plt

def arcLength (array):
    c = 0
    for i in range(1,array.shape[0]):
        distance = np.sqrt ((array[i][0][0]-array[i-1][0][0])**2
                            + (array[i][0][1]-array[i-1][0][1])**2)
        c += distance
    distance = np.sqrt ((array[-1][0][0]-array[0][0][0])**2 
                        + (array[-1][0][1]-array[0][0][1])**2)
    c += distance
    return c

def rdp (array, epsilon):
    dmax = 0
    index = 0
    end = array.shape[0]-1
    for i in range(1, end):
        d1 = np.linalg.norm (np.cross(array[end] - array[1], array[1] - array[i]))
        d2 = np.linalg.norm (array[end] - array[1])
        d = d1 / d2
        if (d > dmax):
            index = i
            dmax = d
        
    ResultList = np.array([])
    
    if (dmax >= epsilon):
        recResults1 = rdp (array[ : index+1], epsilon)[ : -1]
        recResults2 = rdp (array[index : ], epsilon)
        ResultList = recResults1 + recResults2
    else:
        ResultList = [array[1]] + [array[-1]]

    return ResultList


def detectShapes(img, epsilonMultiplier):
    textFont = cv2.FONT_HERSHEY_COMPLEX
    textSize = 1
    textColor = (0, 0, 255)
    textBoldness = 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold = cv2.adaptiveThreshold(gray,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,6)
    threshold2 = cv2.adaptiveThreshold(gray,256,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,6)

    height, width = threshold.shape
    for x in range(height):
        if threshold[x,0] != 256:
            cv2.floodFill(threshold, None, (0, x),0)
        if threshold[x, width-1] != 256:
            cv2.floodFill(threshold, None, (width-1, x),0)
    for y in range(width):
        if threshold[0,y] != 256:
            cv2.floodFill(threshold, None, (y,0),0)
        if threshold[height-1, y] != 256:
            cv2.floodFill(threshold, None, (y, height-1), 0)

    threshold2 = cv2.bitwise_not(threshold2)
    threshold = threshold + threshold2
    
    contours, _ = cv2.findContours (threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000 and area<500000 :
            res = rdp (cnt, epsilonMultiplier * arcLength(cnt)) 
            approx = np.asarray (res)[ : -1]
            #uncomment below for drawing countours
            #cv2.drawContours(img, [approx], 0, (0,0,255), 10)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if len(approx) == 3:
                cv2.putText (img, "Triangle", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            elif len(approx) == 4:
                cv2.putText (img, "Rectangle", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            elif len(approx) == 5:
                cv2.putText (img, "Pentagon", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            elif len(approx) == 6:
                cv2.putText (img, "Hexagon", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            elif len(approx) == 10:
                cv2.putText (img, "5-Star", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            elif len(approx) == 12:
                cv2.putText (img, "6-Star", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)
            else:
                cv2.putText (img, "Circle", (x, y), 
                             textFont, textSize, 
                             textColor, textBoldness)

	
def main():
	img1 = cv2.imread ("input_images/i1.jpg")
	img2 = cv2.imread ("input_images/i2.jpg")
	img3 = cv2.imread ("input_images/i3.jpg")
	img4 = cv2.imread ("input_images/i4.jpg")
	img5 = cv2.imread ("input_images/i5.jpg")
	detectShapes (img1, 0.024)
	detectShapes (img2, 0.024)
	detectShapes (img3, 0.024)
	detectShapes (img4, 0.024)
	detectShapes (img5, 0.024)
	cv2.imwrite ('output_images/i1_output.jpg', img1) 
	cv2.imwrite ('output_images/i2_output.jpg', img2) 
	cv2.imwrite ('output_images/i3_output.jpg', img3) 
	cv2.imwrite ('output_images/i4_output.jpg', img4) 
	cv2.imwrite ('output_images/i5_output.jpg', img5) 
	

if __name__ == '__main__':
    main()

