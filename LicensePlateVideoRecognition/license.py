'''
- Extract license plate from image via computer vision techniques
- Apply optical character recognition to determine plate number
'''

import cv2
import numpy as np
import imutils
import os
import tensorflow as tf
from skimage.filters import threshold_local
from skimage import measure

'''
----sortContours()-----
INPUT: character contours []
OUTPUT: sorted character contour []
PROCESS:
1. find bounding boxes for each character contour in list
2. Zip bbs and ccs together using lambda for matching
'''
def sortContours(cc):
    i = 0
    # find bounding boxes for each character contour in list
    bounding = [cv2.boundingRect(c) for c in cc]
    (cc, bounding) = zip(*sorted(zip(cc,bounding), key=lambda b: b[1][i], reverse=False))
    return cc

'''
----segmentChars()-----
INPUT: plate image, fixedWidth
OUTPUT: sorted character contour []
PROCESS:
- extract Value channel from HSV format of image
- get characters on plate via adaptive thresholding
- resize img & threshold to fixed width (orig size)
- do connected component analysis
- init mask to store possible character locations
'''
def segmentChars(plateImg,fixedWidth):
    V = cv2.split(cv2.cvtColor(plateImg, cv2.COLOR_BGR2HSV))[2]

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    threshold = cv2.bitwise_not(threshold)

    # resize license plate to actual size
    plateImg = imutils.resize(plateImg, width=fixedWidth)
    threshold = imutils.resize(threshold,width=fixedWidth)
    # convert threshold to greyscale
    greyThreshold = cv2.cvtColor(threshold,cv2.COLOR_GRAY2BGR)

    # connected component analysis
    labels = measure.label(threshold,background=0)

    # init mask to store locations of candidate chars
    canChars = np.zeros(threshold.shape,dtype='uint8')

    # loop over unique components
    chars = []
    for label in np.unique(labels):
        # if its a background label, ignore it
        if label==0:
            continue
        # else: make label mask to display the connected components for current label
        mask = np.zeros(threshold.shape,dtype='uint8')
        mask[labels==label] = 255

        # find contours of label mask
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]

        # check that theres at least 1 contour found
        if len(contours)>0:
            # get largest contour
            c = max(contours, key=cv2.contourArea)
            # get bounding box for largest contour
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # calculate aspect ratio, solodity, height ration
            aspectRatio = boxW/float(boxH)
            solidity = cv2.contourArea(c)/float(boxW*boxH)
            heightRatio = boxH/float(plateImg.shape[0])

            # check if aspect < 1.0, solidity > 0.15 & height between 0.5 and 0.95
            if aspectRatio < 1.0 and solidity > 0.15 and heightRatio > 0.5 and heightRatio <0.95 and boxW > 14:
                # calculate convex outside of contour and draw it on char masks
                hull = cv2.convexHull(c)
                cv2.drawContours(canChars, [hull],-1,255,-1)
        contours, hr = cv2.findContours(canChars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # sort them
            contours=sortContours(contours)
            # add pixels to dimensions of char
            pixels = 4
            for c in contours:
                (x,y,w,h) = cv2.boundingRect(c)
                if y > pixels:
                    y = y-pixels
                else:
                    y = 0
                if x > pixels:
                    x = x-pixels
                else:
                    x = 0
                temp = greyThreshold[y:y+h+(pixels*2),x:x+w+(pixels*2)]
                chars.append(temp)
                return chars
        else:
            return None
        
'''
CLASS: PlateFinder
METHODS:
- __init__: builds object with min and max size, and rectangle element structure

- preprocess(self, inputImg): gaussian blur image, convert img to grayscale, use sobelX to get vertical edges,
    set otsu thresholding, redefine PlateFinder morphology

- extractContours(self, postProcessImg): find & return external contours from preprocessed img

- findPossiblePlates(self, inputImg): 

- findCharsOnPlate(self, plate):

- cleanPlate(self, plate):

- checkPlate(self, inputImg, contour):

- ratioCheck(self, area, width, height): 

- preRatioCheck(self, area, width, height):

- validateRatio(self, rect):

- 
'''
class PlateFinder:
    # constructor
    def __init__(self, minPlateArea, maxPlateArea):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure=cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(22,3))

    '''
    ----preprocess()-----
    INPUT: inputImage
    OUTPUT: close Morphed image
    PROCESS:
        1. Apply gaussian blur to img
        2. convert img to greyscale
        3. Get vertical edges via sobelX
        4. Find threshold of vertical edge image
        5. Close morph threshold image
    '''
    def preprocess(self, inputImg):

        # Add gaussian blur to inputImg
        blurred = cv2.GaussianBlur(inputImg, (7,7),0)

        # Convert image to greyscale
        grey = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)

        # Apply sobelX to get vertical edges 
        sobelX = cv2.Sobel(grey,cv2.CV_8U,1,0,ksize=3)

        # Apply otsu's thresholding to find threshold of vertical edged image
        ret2, thresholdImg = cv2.threshold(sobelX, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Close Morph the thresholded image & return it
        element = self.element_structure
        morphed = thresholdImg.copy()
        cv2.morphologyEx(src=thresholdImg,op=cv2.MORPH_CLOSE, kernel=element,dst=morphed)
        return morphed
    
    '''
     ----extractContours()-----
    INPUT: image after preprocessing
    OUTPUT: identified contours
    PROCESS:
    
    '''
    def extractContours(self, img):
        contours, _ = cv2.findContours(img,mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        return contours
    
    '''
     ----cleanPlate()-----
    INPUT: plate
    OUTPUT: plate, T/F, bounding box
    PROCESS:
    - apply image segmentation by extracting value channels from HSV format of plateImg
    - binarize plate img via adaptive thresholding
    - bitwise_not on img to find connected components and get candidate chars
    '''
    def cleanPlate(self, plate):
        # convert plate to greyscale
        grey  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # set adaptive threshold to binarize plate
        threshold = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # find all contours
        contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if there are contours 
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            # get the index of the largest one in the area
            maxIndex = np.argmax(areas)
            maxContourArea = areas[maxIndex]
            # get bounding rectangle
            x,y,w,h = cv2.boundingRect(maxContourArea)
            rect = cv2.minAreaRect(maxContourArea)
            
            # if the ratio doesnt match
            if not self.ratioCheck(maxContourArea, plate.shape[1],plate.shape[0]):
                return plate, False, None
            # else if it does match, return plate, true & bounding box
            return plate, True, [x,y,w,h]
        # if there are no contours
        return plate, False, None

    '''
     ----ratioCheck()-----
     INPUT: area, width, height
     OUTPUT: T/F if ratio is valid
     PROCESS:
     - calculate ratio & use predetermined ratio min, max to determine if ratio is valid
    '''
    def ratioCheck(self, area, width, height):
        # get min and max area
        min = self.min_area
        max = self.max_area

        # ratio min & max
        rMin = 3
        rMax = 6

        # calc ratio
        ratio = float(width)/float(height)
        if ratio < 1:
            ratio = 1/ratio
        if (area < min or area > max) or (ratio < rMin or ratio>rMax):
            return False

        return True
    
    '''
     ----preRatioCheck()-----
     INPUT: area, width, height
     OUTPUT: T/F if ratio is valid
     PROCESS:
     - calculate ratio & use predetermined ratio min, max to determine if ratio is valid
    '''
    def preRatioCheck(self, area, width, height):
          # get min and max area
        min = self.min_area
        max = self.max_area

        # ratio min & max
        rMin = 2.5
        rMax = 7

        # calc ratio
        ratio = float(width)/float(height)
        if ratio < 1:
            ratio = 1/ratio
        if (area < min or area > max) or (ratio < rMin or ratio>rMax):
            return False

        return True

    '''
     ----validateRatio()-----
     INPUT: rect
     OUTPUT: T/F if ratio is valid
     PROCESS:
     - check if angle, height & width, area are valid
    '''
    def validateRatio(self, rect):
        # get parts of rect
        (x,y), (width,height), rAngle = rect
        # check if angle is valid
        if width > height:
            angle = -rAngle
        else:
            angle = 90 + rAngle
        if angle > 15:
            return False
        # check if height & width are both valid
        if height==0 or width==0:
            return False
        # check if area is valid via preRatioCheck
        area = width*height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True

    '''
     ----checkPlate()-----
     INPUT: input image, contour
     OUTPUT: img after all processing & checks, chars on plate, plate coords or NONE
     PROCESS:
     - check if ratio of contour is valid
     - get bounding box 
     - create post-ratio validation img
     - clean the plate with cleanPlate()
     - if there was a plate found, find the chars with findCharsOnPlate()
     - if 8 chars were found on plate, get the coords and return
    '''
    def checkPlate(self, img, contour):
        # get the min rect of the contour
        minRect = cv2.minAreaRect(contour)

        # if ratio of minRect is valid
        if self.validateRatio(minRect):
            # get bounding box
            x,y,w,h = cv2.boundingRect(contour)
            afterValidImg = img[y:y+h, x:x+w]
            
            #clean plate
            afterCleanImg, plateFound, coords = self.cleanPlate(afterValidImg)
            
            # if there was a plate found
            if plateFound:
                # find the chars on the plate
                chars = self.findCharsOnPlate(afterCleanImg)

                # if chars were found on the plate and there are 8 of them
                if chars is not None and len(chars == 8):
                    # get coords
                    xx,yy,ww,hh = coords
                    coords = xx+x, yy+y
                    afterCheckPlateImg = afterCleanImg

                    return afterCheckPlateImg, chars, coords
        return None, None, None
    '''
     ----findPossiblePlates()-----
     INPUT: input image
     OUTPUT: list of possible plates or None
     PROCESS:
     - preprocesses img with preprocess()
     - check side ratios and areas of all contours
     - cleans img inside contours with checkPlate() & cleanPlate()
     - find all characters on plate 
    '''
    def findPossiblePlates(self, img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        # preprocess img
        self.after_preprocess = self.preprocess(img)

        # find possible plate contours after img preprocess
        possiblePlateContours = self.extractContours(self.after_preprocess)

        # go thru all possible contours & check plates
        for c in possiblePlateContours:
            plate, chars_on_plate, coords = self.checkPlate(img, c)
            # if there is a plate, add it to list of possible plates
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(chars_on_plate)
                self.corresponding_area.append(coords)
        if len(plates)>0:
            return plates
        else:
            return None

    
    '''
     ----findPossiblePlates()-----
     INPUT: plate
     OUTPUT: list of characters found on given plate
     PROCESS: calls segmentChars()
    
    '''
    def findCharsOnPlate(self, plate):
        result = segmentChars(plate, 400)
        if result:
            return result

if __name__=='__main__':
    # find the plate using FindPlate object
    plate = PlateFinder(minPlateArea=4100,maxPlateArea=15000)

    # create OCR model
    
    # define video capture for testing