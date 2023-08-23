# importing librraies 

import argparse 
from shadow_highlight_correction import correction
import math
import os
import re
from PIL import Image
import pandas as pd 
import imutils
import cv2 as cv
import variable
import cv2
import pickle
import numpy as np

#GOAL 1: score function which scores ing pppln according to how well it perfomed  ✔

#Result: does run, but 't detect images BUT needs a confusion matrix, grid search and laod images in batches (as for 2K dataset) ? FIXED

#2: takes both 2dprinted tags(test) and 3d printed tags and detects the id's correctl, scores iT ✔

#3 : check the datatype whch is return by the function, rewrite all the fucntion manually - Done - ✔ 
# (Original id return array of ints, predict ints array return arr of int in array - [[1]])

#4: check/score 2 pplns - Fahds (4.1) - ✔ & Amrit (4.2) - ✔

#5 : Implement My, F, A pipelines as an option in the command line - ✔

#6: Build a better datset, test pplns on it 2500 ✔

#7: Dataset from different angles/calculating the distance - ✔

#  image processing in batches, implement and rewrite functions - ✔

# NEEDS TO BE DONE 7: Implement (precision and recall) - ✔

#8 Predicts 2 tags, return 2 tags  

#9 function that takes all the pplns 

#10 saves the result to the file 

#11 TASKS for July 11 rewite function for 2ID just for a 1 set 
#Original Id and scoring functim 

#12 #Function: inserting  any pipelines and getting the result 

##CLean the code rn - repeating values, implenet 2 id's to every ppln 
# Score the string and add the classification to the scoring function
# label each predicted id as option(FP,TP,TN,FN), compere to exel while manually created 

#I was trying to import functions but I got work more of functions I importing like these which imports images 
#from Amrits_ppln import correction , contrast, threshold, calibration_frame, clahe, blur, THRESHOLD_PX_VAL 



directory  = r'D:\AI research internship\opencv_scripts\2_id'
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)

#Functios for implenattion of the scring fucntion in array format --------------------------------------------------------------------
#load images in image dictionary - small size datasets only (what I inittially started with)
def load_images(directory):
        image_dict = {}
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                try:
                    image = cv2.imread(image_path)
                    image_dict[filename] = image
                except OSError:
                    print(f"Unable to open image: {filename}")
        return image_dict

import os
import cv2



#using regular expression to extratct id from images names, and add it to a string - Test data 
def original_id(image_dict):
    digit_array = []
    pattern = r'^\d{1,2}'
    
    for key in image_dict.keys():
        match = re.match(pattern, key)
        if match:
            first_digits = int(match.group())
            digit_array.append(first_digits)
    print(digit_array)        
    return digit_array


#3 CREATE an ARRAY with Tags that were predicted - ✔
#using regular expression to extratct id from images names, and add it to a string - Test data 
def My_ppln(image_dict):
    p_id_arr = []
    for ids, img in image_dict.items():
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clache = cv2.createCLAHE(clipLimit=40)
        frame_clache = clache.apply(gray)           
        th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 1)
        blurred = cv2.GaussianBlur(th3, (21, 21), 0)
        flipped = cv2.flip(blurred, 1)
        _, ids, _ = detector.detectMarkers(flipped)
        p_id_arr.append(ids)

    print('My array with predicted id is: ', p_id_arr)
    return p_id_arr

#1/3 A_ppln for the Array format 
def A_ppln_2(image_dict):
            arr = []
            p_id_set = set()
            for ids, img in image_dict.items():
                img_corrected = correction(img, 0, 0, 0, 0.6, 0.6, 30, .3)
                img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
                if calibration_frame is not None:
                    img_norm = img_gray - calibration_frame
                else:
                    img_norm = img_gray

                img_contrast_enhanced = contrast(img_norm, clahe)
                img_blurred = blur(img_contrast_enhanced, (5, 5))
                img_thresholded = threshold(img_blurred, THRESHOLD_PX_VAL)
                flipped = cv2.flip(img_thresholded, 1)
                ids = A_detect(flipped)
                
                p_id_set = set()
                
                if ids is not None:
                    for inner_arr in ids:
                        for i in inner_arr:
                            p_id_set.add(i)
                        
                arr.append(p_id_set) 
            return arr
        
# F_ppln CREATE an ARRAY with Tags that were predicted - ✔ 2/4 

def F_ppln(image_dict):
    p_id_arr = []
    for ids, images in image_dict.items():
        transformation = cv2.cvtColor((images), cv2.COLOR_BGR2GRAY)
        transformation = cv2.bitwise_not(transformation)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        transformation = clahe.apply(transformation)
        transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
        _, transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
        # flipped = cv2.flip(transformation, 1)
        _, ids, _ = detector.detectMarkers(transformation)
        p_id_arr.append(ids)


        
    
    print('Fahads array with the predicted id is: ', p_id_arr)
    return p_id_arr


def calc_p_r(original_ids, predicted_ids):
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(original_ids)):
        if original_ids[i] == predicted_ids[i]:
            true_positive += 1
        else:
            false_negative += 1

    false_negative = len(original_ids) - true_positive

    precision = 0
    recall = 0

    if true_positive + false_positive != 0:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative != 0:
        recall = true_positive / (true_positive + false_negative)

    return precision, recall


def score(original_id, predicted_id):
    scores = 0
    total = len(original_id)
    predicted_id_count = len(predicted_id)
    print()
    print('Total images:', total)
    print('Predicted images:', predicted_id_count)

    for id in original_id:
        if isinstance(id, int) and id in predicted_id:
            scores += 1

    precision, recall = calc_p_r(original_id, predicted_id)
    ratio = (scores / total) * 100
    print('Scores:', scores)
    return scores, total, ratio, precision, recall

#7 Display ratio based on the precious scoring subfunction  - ✔

#TO ADD: toal FP/TN/FN/TP - Array format
def info(score,total,ratio,precision, recall):
     total = total
     score = score
     ratio = ratio
     precision=precision
     recall = recall
     print(f"Image processing pipeline scored at {int(ratio)} %")
     print(f"Out of {total} images {score} were predicted")
     print('Score:', ratio, '%')
     print('precision:', precision, '%')
     print('recall:', recall, '%')

# Functions for finding distance/angle/brightness ------------------------------------------------------------------------------------------------------------

KNOWN_DISTANCE = 11
  
KNOWN_WIDTH = 3


def distance_to_camera(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth



def find_marker(img):
            # Convert the image to grayscale, blur it, and detect edges
            transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            transformation = cv2.bitwise_not(transformation)
            clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
            transformation = clahe.apply(transformation)

            transformation = cv2.GaussianBlur(transformation, (21, 21), 0)

            transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)

            _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
            edged = cv2.Canny(transformation, 35, 125)

            # Find the contours in the edged image and keep the largest one;
            # we'll assume that this is our piece of paper in the image
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # Find the contour with the maximum area
            c = max(cnts, key=cv2.contourArea)
            marker = cv2.minAreaRect(c)


            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            print('Marker  is', marker, type(marker))
            return marker
# Fads ppln finding marker , for the distance between a marker and camera 
def find_marker_2(img):
        transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        transformation = cv2.bitwise_not(transformation)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        transformation = clahe.apply(transformation)

        transformation = cv2.GaussianBlur(transformation, (21, 21), 0)

        transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)

        _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
        corners, ids, rejected = detector.detectMarkers(transformation) 
        detected_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
        # Calculate the width of the ArUco tag
        
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (7, 7)
        zeroZone = (1, 1)

        refined_corners = []
        for corner_set in corners:
            refined = cv2.cornerSubPix(transformation, corner_set, winSize, zeroZone, criteria)
            refined_corners.append(refined)
            print('refined corner is ', refined_corners)

        visualizer = cv2.cvtColor(transformation, cv2.COLOR_GRAY2BGR)
        
        # Calculate the width of the ArUco tag
        # Calculate the width of the ArUco tag
        if len(refined_corners) > 0:
            corner_points = refined_corners[0][0]  # Extract corner points
            width_pixels = np.linalg.norm(corner_points[0] - corner_points[2])
        else:
            width_pixels = 0

        # Draw the detected markers on the visualizer
        visualizer = cv2.aruco.drawDetectedMarkers(visualizer, refined_corners, ids) 
        print('Marker for arUco is', detected_markers, type(detected_markers))
        print('Marker for arUco width is', width_pixels, type(width_pixels))
        
        return  ids, corners




def isbright(image, dim=10, thresh=0.5):
        # Resize image to 10x10
            image = cv2.resize(image, (dim, dim))
            # Convert color space to LAB format and extract L channel
            L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
            # Normalize L channel by dividing all pixel values with maximum pixel value
            L = L/np.max(L)
            value = np.mean(L) 
            # Return True if mean is greater than thresh else False
            if np.mean(L) > thresh:
                return  value
            else:
                return   value
        # compute the bounding box of the of the paper region and return it

#0. Function that Generates dictionary with images from the file - ✔
# Key: 9_angle_6_.png, Value: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x178C0FF6BF0>

#------------------------------------------------------------------------------------------------------------------------------------------

#1 Loads images in batches Fucntion that I used to load Big datasets (Currently)

def load_images_in_batches(directory, batch_size, batch_index=0):
    # os - intereacting with the operating system
    # listdir - returns a list of all the files
    image_files = os.listdir(directory)
    total_images = len(image_files)
    #math.ceil() function is used to round up to the nearest integer value
    num_batches = math.ceil(total_images / batch_size)

    for i in range(num_batches):
        batch_index +=1
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_images)
        batch_files = image_files[start_index:end_index]
        
        batch_images = {}
        for file in batch_files:
            
            image_path = os.path.join(directory, file)
            image = cv2.imread(image_path)
            image_name = os.path.splitext(file)[0]  # Extract name without extension
            batch_images[image_name] = image
        print()
        print('Batch number: ', batch_index)
        print()
        
        yield batch_images
        
        

#2 CREATE an ARRAY with Tag original ID's by parcing the key string, as getting first digits of it and save to an array 'Original ID' - Key: 9_angle_6_.png - > 9

#check type - ✔ returns array of ints

#Info: Original id's of the images which is an array of ints  

# Scoring function implented in array format ---------------------------------------------------------------------------------


def pplns(ppln, directory):
 batch_size = 150
 for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = ppln(batch)
       
       # int(x[0, 0])
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]

        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)

# A_ppln CREATE an ARRAY with Tags that were predicted - > reimplemnt function with array of sets 

calibration_frame = None

THRESHOLD_PX_VAL = 100

CLIP_LIMIT = 20.0 

def contrast(image, clahe):
    clahe_out = clahe.apply(image)
    return clahe_out

def blur(image, kernel):
    return cv2.blur(image, kernel)

def threshold(image, px_val):

    thresholded = cv2.adaptiveThreshold(image, 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 
                                        89, 
                                        2)
    return thresholded


def drawMarkers(img, corners, ids, borderColor=(255,0,0), thickness=25):
    if ids == []:
        ids = ["R"] * 100
    for i, corner in enumerate(corners):
        if ids[i] == 17:
            continue
        corner = corner.astype(int)
        cv2.line(img, (corner[0][0][0], corner[0][0][1]), (corner[0][1][0], corner[0][1][1]), borderColor, thickness)
        cv2.line(img, (corner[0][1][0], corner[0][1][1]), (corner[0][2][0], corner[0][2][1]), borderColor, thickness)
        cv2.line(img, (corner[0][2][0], corner[0][2][1]), (corner[0][3][0], corner[0][3][1]), borderColor, thickness)
        cv2.line(img, (corner[0][3][0], corner[0][3][1]), (corner[0][0][0], corner[0][0][1]), borderColor, thickness)
        cv2.putText(img, str(ids[i]), (corner[0][0][0], corner[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
    return img

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(5, 5)) 
def invert(img):
    image_not = cv2.bitwise_not(img)
    return image_not


def A_detect(image, draw_rejected = False):
    (corners, ids, rejected) =  detector.detectMarkers(image)
    (corners_inv, ids_inv, rejected_inv) =  detector.detectMarkers(invert(image))
    (corners_hflip, ids_hflip, _) = detector.detectMarkers(invert(cv2.flip(image,1)))

    back_to_color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    if draw_rejected:
        detected = drawMarkers(back_to_color.copy(), rejected, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), rejected_inv, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
    else:
        detected = drawMarkers(back_to_color.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
        detected = drawMarkers(cv2.flip(detected.copy(), 1), corners_hflip, ids_hflip, borderColor=(83, 235, 52))


    return ids_hflip




def A_ppln(image_dict):
    p_id_arr = []
    for ids, img in image_dict.items():
         
        img_corrected = correction(img, 0, 0, 0, 0.6, 0.6, 30, .3)
        img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
        if calibration_frame is not None:
            img_norm = img_gray - calibration_frame
        else:
            img_norm = img_gray

        img_contrast_enhanced = contrast(img_norm, clahe)
        img_blurred = blur(img_contrast_enhanced, (5, 5))
        img_thresholded = threshold(img_blurred, THRESHOLD_PX_VAL)
        flipped = cv2.flip(img_thresholded, 1)
        ids = A_detect(flipped)
        p_id_arr.append(ids)
    print('Amrits array with predicted id is: ', p_id_arr)
    return p_id_arr

#4


#5 Function that combines and displays 2 images side by side  - ✔


def combined_2(img1, img2):


 height = max(img1.shape[0], img2.shape[0])
 img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
 img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

# Create a new image with double width
 combined_image = Image.new("RGB", (img1.shape[1] + img2.shape[1], height))

# Paste the images side by side
 combined_image.paste(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), (0, 0))
 combined_image.paste(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), (img1.shape[1], 0))

 return combined_image

#6 Calculates the precison and recall - ✔ Array format 
#if original is empty set(), and predicted is empty TN++  EX:  (set() - > set()) - TN++
#id the original is empty set(, and predicted is something else then empty FP++ EX: (set() -> {17}) - FP++ 
#ALso if the set is one value, and th epredicted is anothe value, FP++ EX: ({23,40} -> {17}) - FP++ 



# Handle All the scores, including renewed datasets, and F new ppln -----------------------------------------------

def load_images(directory):
        image_dict = {}
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                try:
                    image = cv2.imread(image_path)
                    image_dict[filename] = image
                except OSError:
                    print(f"Unable to open image: {filename}")

        return image_dict




def clean_string(string):
        digits = re.findall(r'\d+', string)
        if len(digits) > 1:
            digits = digits[:-1]  
        cleaned_string = '_'.join(digits)
        count = len(digits)
        
        return count, cleaned_string



def original_id_2(image_dict):
        arr = []

        for key in image_dict.keys():
            count, key = clean_string(key)
            #print('Count of ids is', count, 'the key is', key)
            pattern = r'^(\d{1,2})'  # Create the pattern based on the ID count

            for _ in range(1, count):
                pattern += r'_(\d{1,2})'

            digit_set = set()
            match = re.match(pattern, key)
            if match:
                for group in match.groups():
                    digit_set.add(int(group))
                arr.append(digit_set)
            else:
                print(f"Key '{key}' does not match the pattern.")
        print('Original ids')        
        print(arr)
        return arr




    
def F_2_ppln(image_dict):
            arr = []
            p_id_set = set()
            for ids, img in image_dict.items():
                transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                transformation = cv2.bitwise_not(transformation)
                clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
                transformation = clahe.apply(transformation)

                transformation = cv2.GaussianBlur(transformation, (21, 21), 0)

                transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)

                _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)  # Renamed 'transformation' to 'de
                _,ids,_ = detector.detectMarkers(transformation)   
                p_id_set = set()
                
                if ids is not None:
                    for inner_arr in ids:
                        for i in inner_arr:
                            p_id_set.add(i)
                        
                arr.append(p_id_set)
            print('F_2 predictions: ')    
            print(arr) 
            return arr

def calc2(original_ids, predicted_ids):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        FP = 0
        FP2=0
        scores = 0
        intersection = 0
        total = len(original_ids)
        for set_o, set_p in zip(original_ids, predicted_ids):
          #1 value and not empty cases part-1
          #if len(set_o)==1:
            #True positive 
            if set_o != set():
                intersection = set_o&set_p
                TP+=len(intersection)
                if len(intersection)>0:
                    scores += 1
              #false positive one {40}->{13}
                if set_o != set_p:
                    if set_p != set():
                        FP += 1
              # false negative {20}->set
                if set_p == set():
                    FN += 1
            #empty set_o part 2 
            if set_o == set():
               #True negative case set() -> set()
               if set_o == set_p:
                    TN += 1
                    scores += 1
                # set() - > {17}
               if set_o != set_p:
                    FP2 += 1
        #total_TP, total_FP, total_FN, true_negative, false_positive2, scores, total
        return TP, TN, FP, FN, FP2, scores, total   
    
    #return array with classification TP,FP,FN,TN [[0,1,0,0],[1,0,0,0]]
    #total_TP, total_FP, total_FN, true_negative, false_positive2, scores, total
    #TO DO: recheck if it produces correct output 
    
def class_arr(original_ids, predicted_ids):
            
            true_negative = 0
            

            classification = []


            for set_o, set_p in zip(original_ids, predicted_ids):

                if set_o != set():  # Multiple cases for TP, FN, FP
                    inner=[]
                    true_negative = 0
                    intersection = set_o & set_p
                    true_positive = len(intersection)
                    #true_positive append
                    #print('true_positive', true_positive)
                    inner.append(true_positive)
                    false_negative = len(set_p - set_o)
                    #false_negative
                    #print('false_negative', false_negative)
                    inner.append(false_negative)

                    false_positive = len(set_o) - true_positive - false_negative
                    #print('false_positive', false_positive)
                    #false_positive
                    inner.append(false_positive)
                    true_negative = 0
                    inner.append(true_negative)
                    #print('true_negative', true_negative)
                    classification.append(inner)


                if set_o==set():
                    if set_o==set_p:
                        true_negative = 1
                        inner.append(true_negative)
                        true_positive=0
                        inner.append(true_positive)
                        false_negative=0
                        inner.append(false_negative)
                        false_positive=0
                        inner.append(false_positive)

                    # Calculate score for each true positive


            return classification
        
    

  #Format of function info looks like - Predicted: Score:11 | TP:11, FN:4, TN:0, FP-1:135, FP-2:0 | precision:0, recall:0.0
    
def info(TP, TN, FP, FN, FP2, scores, total):
        precision = 0
        recall = 0

        if TP + FP + FP2 != 0:
            precision = TP / (TP + FP + FP2)
        else:
            precision = 0

        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        print(f'Predicted: Score:{(scores/total)*100}% | TP:{TP}, FN:{FN}, TN:{TN}, FP1:{FP}, FP2:{FP2} | precision:{precision}, recall:{recall}')
        print(f'Out of {total} images, {TP} were predicted accurately')


  # save values to a file
   

#convert 2 array with sets to array with strings 
def convert(original_ids, predicted_ids):
        # Find the maximum set length in original_ids and predicted_ids
        max_length = max(max(len(s) for s in original_ids), max(len(s) for s in predicted_ids))

        # Fill sets with missing elements with None
        original_ids_filled = [list(s) + [None] * (max_length - len(s)) for s in original_ids]
        predicted_ids_filled = [list(s) + [None] * (max_length - len(s)) for s in predicted_ids]

        # Convert sets to strings
        original_ids_str = [', '.join(map(str, s)) for s in original_ids_filled]
        predicted_ids_str = [', '.join(map(str, s)) for s in predicted_ids_filled]

        # Create a DataFrame with the 'IDs', 'Predictions', and 'Batch' columns
        return original_ids_str, predicted_ids_str


def cvs(original_ids, predicted_ids,  batch):
        # Create a DataFrame with the 'IDs', 'Predictions', and 'Batch' columns
        df = pd.DataFrame(columns=['IDs', 'Predictions', 'Batch'])

        # Convert empty sets to None
        original_ids_1, predicted_ids_1  = convert(original_ids,predicted_ids)

        #classification arr

        classification = class_arr(original_ids,predicted_ids)

        # Create DataFrame from the second function's output
        df_classification = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])

        # Fill the DataFrame with data from original_ids and predicted_ids
        batch_n = 1
        for original_id, predicted_id in zip(original_ids_1, predicted_ids_1):
            df = pd.concat([df, pd.DataFrame({'IDs': [original_id], 'Predictions': [predicted_id], 'Batch': [batch_n]})], ignore_index=True)
            batch_n+=1
        # Concatenate both DataFrames side by side
        result_df = pd.concat([df, df_classification], axis=1)
# General fucntion ppln for img in batches 
 # TASK: add original, predicted , classification, batch_number to the dicitonary - > write dictionary to a cvs file  
def pplns_batch(ppln, directory):
        batch_size = 150
        for batch in load_images_in_batches(directory, batch_size):
            original_ids = original_id_2(batch)
            #print(original_ids)
            predicted_ids = ppln(batch)
            cvs(original_ids,predicted_ids, batch)
            #new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
            #print(predicted_ids)
            TP, TN, FP, FN, FP2, scores, total = calc2(original_ids,predicted_ids)
            info(TP, TN, FP, FN, FP2, scores, total)
 # ------------------------------------------------------------------------------------------



cap = cv2.VideoCapture(0)


#Command_Line -----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--my_score', action='store_true')
parser.add_argument('--Fahad_score', action='store_true')
parser.add_argument('--Amrit_score', action='store_true')
parser.add_argument('--score_2id', action='store_true')
parser.add_argument('--score_all_cases', action='store_true')
parser.add_argument('--distance', action='store_true')
parser.add_argument('--distance_tag', action='store_true')
parser.add_argument('--angle', action='store_true')
parser.add_argument('--feedback_all', action='store_true')
parser.add_argument('--empty', action='store_true')

args = parser.parse_args()

output_filename = 'output.doc'  
# Save the output image to a file


if args.my_score:

    batch_size = 150
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'


  
    
    
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = My_ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)


    print('My ppln score: ')   
    print('')
    
    print("Batch complete")


#F_ppln with args - ✔


if args.Fahad_score:
   
  
    batch_size = 150
  
  
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'
    print('Fahd ppln score: ')   
    print('')
    
    pplns(F_ppln)
     


# A ppln ---------------------------------------------------------------------------------------------------------------------------------
    if args.Amrit_score:
        
        batch_size = 150
        directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'

        print('Amrit ppln score: ')  
        print('')
    
        
        pplns(A_ppln)

# The End of A ppln ------------------------------------------------------------------------------------------------------------------------

# Testing 2ID's dataset ----------------------------------------------------------------------------------------------------------------------- 

if args.score_2id:
    
    
    print('score for 2 ids')
    directory  = r'D:\AI research internship\opencv_scripts\2_id'

    img_dict = load_images(directory) 

    def original_id_2(image_dict):
        
        arr = []
        
        pattern = r'^(\d{1,2})_(\d{1,2})'

        for key in image_dict.keys():
            digit_set = set()
            match = re.match(pattern, key)
            if match:
                first_digits = int(match.group(1))
                second_digits = int(match.group(2))
                digit_set.add(first_digits)
                digit_set.add(second_digits)
                arr.append(digit_set)
            else:
                print(f"Key '{key}' does not match the pattern.")
        

        print(arr)
        return arr

    def A_ppln_2(image_dict):
        arr = []
        p_id_set = set()
        for ids, img in image_dict.items():
            img_corrected = correction(img, 0, 0, 0, 0.6, 0.6, 30, .3)
            img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
            if calibration_frame is not None:
                img_norm = img_gray - calibration_frame
            else:
                img_norm = img_gray

            img_contrast_enhanced = contrast(img_norm, clahe)
            img_blurred = blur(img_contrast_enhanced, (5, 5))
            img_thresholded = threshold(img_blurred, THRESHOLD_PX_VAL)
            flipped = cv2.flip(img_thresholded, 1)
            ids = A_detect(flipped)
            
            p_id_set = set()
            
             
            if ids is not None:
                for inner_arr in ids:
                    for i in inner_arr:
                        p_id_set.add(i)
                    
            arr.append(p_id_set)
        print('Array for 2 ids is ', arr)    
        return arr
    
    #A_ppln that handles none values 


     #Add count to each of classification param
     #print oroginal value and predict value and lable as any od 4 values 

    def calc_p_r(original_ids, predicted_ids):
        #I did - > it predicted 
        true_positive = 0
        #I did -> it didn't predicted 
        false_negative = 0
        #I didn't do -> it predicted 
        false_positive = 0
        #I didn't do - >  it didn't predict 
        true_negative = 0

        #Q:  do i have to create set with empty values as for possible false positive/true negative cases 

        for i in range(len(original_ids)):
            if original_ids[i] == predicted_ids[i]:
                true_positive += 1
            else:
                false_negative += 1

        false_negative = len(original_ids) - true_positive

        precision = 0
        recall = 0

        if true_positive + false_positive != 0:
            precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative != 0:
            recall = true_positive / (true_positive + false_negative)

        return precision, recall
    

    
 
# ALL cases -------------------------------------------------------------------------------------------------------------------------------------------
if args.score_all_cases:
# Working function that prints clearly putput of scoring function for the 2 datasets -----------------------------------------------------------------------------------------------
#(implement in argline as info score for 2 datasets  -> 41_7)
#issue i am running into none is not itterable - solved - create an empty set instead of none 
#  out of 91 images saves just 2 because sets saves n stores






    directory = r'D:\AI research internship\opencv_scripts\data_set'


    #cleans a string, count ids, ignores the last digit(angle)


    #small dataset 41 img - 1id 
    directory0 = r'D:\AI research internship\opencv_scripts\data_set'
    img_dict = load_images(directory) 
 


   # raturns values, and array with the sets of classification
  

        #print(result_df)

    file_path = 'data999-final0.csv'
        #uncommednt once create new datset
        #result_df.to_csv(file_path, mode='a', index=False, header=not os.path.isfile(file_path))
        #print('Successfully saved to the file!')

#Dataset
    # ! ! ! - I used all the fucntion with pplns function which can be used for any ppln - > Clean up the code, saves time and space *
    #orig_set = original_id_2(img_dict)

    #predict_set = F_2_ppln(img_dict)

    
    #total_TP, total_FP, total_FN, true_negative, false_positive2, scores, total = calc2(orig_set, predict_set)

    #a_orig_set = [set(),{40,30},{5},{40,30},{6},{20,30},set()]

    #a_predicted_set = [set(),{17},{5},{40,30},set(),{20},{5}]

    #total_TP_a, total_FP_a, total_FN_a, true_negative_a, false_positive2_a, scores_a, total = calc2(a_orig_set, a_predicted_set)

    directory1 = r'D:\AI research internship\opencv_scripts\data_set'
    
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'

    #UNcomment for more scores of Scoring function on the different pplns & datasets 
    
    '''

    print()
    print('---------------------------------------------------')
    print('Info for the 7 img  dataset (multiple cases: no ids, 1 id, 2+ ids)')
    print('---------------------------------------------------')
    info(total_TP_a, total_FP_a, total_FN_a, true_negative_a, false_positive2_a, scores_a )
    

    print('---------------------------------------------------')
    print('F-2 ppln | Info for the 41 img dataset (1 case: 1 id)')
    print('---------------------------------------------------')
    print()
    info(total_TP, total_FP, total_FN, true_negative, false_positive2, scores)
'''
    print('---------------------------------------------------')
    print('F-2 ppln | Info for the 3K img dataset (1 case: 1 id - condition: left/rigth )')
    print('---------------------------------------------------')
    pplns(F_2_ppln)

'''

    print('---------------------------------------------------')
    print('A ppln | Info for the 3K img dataset (1 case: 1 id - condition: left/rigth )')
    print('---------------------------------------------------')
    pplns(A_ppln_2)
'''
# ----------------------- Find Distance using Camera Calib 
if args.feedback_all:
        data_1 =  r'D:\AI research internship\opencv_scripts\a_70_d_4'
        data_2 =  r'D:\AI research internship\opencv_scripts\a_70_d_10'
        data_4  = r'D:\AI research internship\opencv_scripts\a_28_d_10'
        data_5  = r'D:\AI research internship\opencv_scripts\a_5_d_4'
        data_6  = r'D:\AI research internship\opencv_scripts\a_50_d_10'
        data_7  = r'D:\AI research internship\opencv_scripts\a_50_d_4'
        data_8 = r'D:\AI research internship\opencv_scripts\a_70_d_20'
        data_9 = r'D:\AI research internship\opencv_scripts\a_50_d_20'
        data_10 = r'D:\AI research internship\opencv_scripts\a_28_d_20'
        data_11 = r'D:\AI research internship\opencv_scripts\a_5_d_20'




        
        
        # 12 - "D:\AI research internship\opencv_scripts\empty_data2"

   # 12 - "D:\AI research internship\opencv_scripts\empty_data2"

        name = {
        data_1: 'a_70_d_4',
        data_2 : 'a_70_d_10',
        data_4: 'a_28_d_10',
        data_5: 'a_5_d_4',
        data_6: 'a_50_d_10',
        data_7: 'a_50_d_4',
        data_8: 'a_70_d_20',
        data_9: 'a_50_d_20',
        data_10 : 'a_28_d_20t',
        data_11: 'a_5_d_20'
    }
        
        path = {
        data_1: 'Angle: 70, Distance: 4, Lighting: low day light',
        data_2 : 'Angle: 70, Distance: 10, Lighting: low day light',
        data_4: 'Angle: 28, Distance: 10, Lighting: low day light',
        data_5: 'Angle: 5, Distance: 4, Lighting: low day light',
        data_6: 'Angle: 50, Distance: 10, Lighting: low day light',
        data_7: 'Angle: 50, Distance: 4, Lighting: low day light',
        data_8: 'Angle: 70, Distance: 20, Lighting: low day light',
        data_9: 'Angle: 50, Distance: 20, Lighting: low day light',
        data_10 : 'Angle: 28, Distance: 20, Lighting: low day light',
        data_11: 'Angle: 5, Distance: 20, Lighting: low day light'
    }


     

        def get_angle_distance(string):
            split_array = string.split('_')
            angle = split_array[1]
            distance = split_array[3]
            
            return angle, distance 
        
        


 
     
#df_distance_angle = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])

        def cvs_all(original_ids, predicted_ids,  folder_name):
                    # Create a DataFrame with the 'IDs', 'Predictions', and 'Batch' columns
                    df = pd.DataFrame(columns=['IDs', 'Predictions', 'Batch'])

                    # Convert empty sets to None
                    original_ids_1, predicted_ids_1  = convert(original_ids,predicted_ids)

                    #classification arr

                    classification = class_arr(original_ids,predicted_ids)

                    # Create DataFrame from the second function's output
                    df_classification = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])
                    # Add Angle and Distance columns using the get_angle_distance function
                    #get the name 
                    angle, distance  = get_angle_distance(folder_name)
                    df_classification['Angle'] = angle
                    df_classification['Distance'] = distance 
                    # Fill the DataFrame with data from original_ids and predicted_ids
                    batch_n = 1
                    for original_id, predicted_id in zip(original_ids_1, predicted_ids_1):
                        df = pd.concat([df, pd.DataFrame({'IDs': [original_id], 'Predictions': [predicted_id], 'Batch': [batch_n]})], ignore_index=True)
                        batch_n+=1
                    # Concatenate both DataFrames side by side
                    result_df = pd.concat([df, df_classification], axis=1)
                    result_df.to_csv(f'final_dataframe_{folder_name}.csv', index=False)
                    print('successfully SAVED')

        def pplns_batch(ppln, directory, folder_name):
                    batch_size = 150
                    for batch in load_images_in_batches(directory, batch_size):
                        original_ids = original_id_2(batch)
                        #print(original_ids)
                        predicted_ids = ppln(batch)
                        cvs_all(original_ids,predicted_ids, folder_name)
                        #new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
                        #print(predicted_ids)
                        TP, TN, FP, FN, FP2, scores, total = calc2(original_ids,predicted_ids)
                        info(TP, TN, FP, FN, FP2, scores, total)

        for value, key in zip(path.values(), path.keys()):
          folder_name = name[key]  # Get the corresponding folder name from the name dictionary
          print()
          print(f'Folder: {folder_name}')
          print()
          print('Conditions:', value)
          print()
          pplns_batch(F_2_ppln, key, folder_name)
            #  
        
        #load all the datasets #300 img each 
        #Score them 
        #print score 
        #creact cvs file with the data and results for all the datsets 
        #visualize 
if args.distance:

   
    #  Distance  -------------------------------------------------------------------------------------------

    cap = cv2.VideoCapture(0)
     

    IMAGE_PATHS = ['(1).jpeg']


    image = cv2.imread(IMAGE_PATHS[0])
    find_marker(image)


    #After running the script Camera_calib.py - the focal lenght I got is: 
    focal_length = int(772.3458251953125) 

    #------------------------------------------------------------------------------------------------------------------------------
    #Start 
    #Track the tag, and calculate the Distance 

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        #frame = cv2.resize(frame, (300, 300))
        
        # Find the marker in the frame
        #marker_a = find_marker_a(frame)
        marker = find_marker(frame)
        print('marker is:', marker)
        
        #Find aruco tag
        
        
        if marker is not None:
         #if marker > 0:
            # Calculate the focal length using the known distance and width
            
            
            # Calculate the distance in inches
            inches = distance_to_camera(KNOWN_WIDTH, focal_length, marker[1][0])

            #brightness

            # Convert 'number' to a formatted string with one decimal place
            formatted_number = "{:.1f}".format(inches)

            bright = isbright(frame)
            
            # Draw a bounding box around the image and display it
            box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (255, 255, 0), 4)
            
            text = f"Distance: {inches}"

                # Put the text on the right side of the frame
            cv2.putText(frame, text ,
                        (200,200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                    (255, 255, 0), 
                    4, 
                    cv2.LINE_4)



        # Display the frame (whether or not a marker is detected)
        cv2.imshow('img_check', frame)

        # Wait for 1000 milliseconds and check if the user pressed 'x' to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #Finding distabce for the tag ------------------------------------------------------------------
if args.distance_tag:

    KNOWN_DISTANCE = 11
    KNOWN_WIDTH = 1


                    

    def find_marker_2(img):
        transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        transformation = cv2.bitwise_not(transformation)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        transformation = clahe.apply(transformation)

        transformation = cv2.GaussianBlur(transformation, (21, 21), 0)

        transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)

        _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
        corners, ids, rejected = detector.detectMarkers(transformation) 
        detected_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
        
        # Calculate the width of the ArUco tag
        
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (7, 7)
        zeroZone = (1, 1)

        refined_corners = []
        for corner_set in corners:
            refined = cv2.cornerSubPix(transformation, corner_set, winSize, zeroZone, criteria)
            refined_corners.append(refined)
            print('refined corner is ', refined_corners)

        visualizer = cv2.cvtColor(transformation, cv2.COLOR_GRAY2BGR)
        
        # Calculate the width of the ArUco tag
        # Calculate the width of the ArUco tag
        if len(refined_corners) > 0:
            corner_points = refined_corners[0][0]  # Extract corner points
            width_pixels = np.linalg.norm(corner_points[0] - corner_points[2])
        else:
            width_pixels = 0

        # Draw the detected markers on the visualizer
        visualizer = cv2.aruco.drawDetectedMarkers(visualizer, refined_corners, ids) 
        print('Marker for arUco is', detected_markers, type(detected_markers))
        print('Marker for arUco width is', width_pixels, type(width_pixels))
        
        return  width_pixels,refined_corners





    def distance_to_camera(knownWidth, focalLength, perWidth):
        focalLength=772
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth


    # initialize the known distance from the camera to the object, which
    # in this case is 11 cm
    KNOWN_DISTANCE = 11
    # initialize the known object width, which in this case, the piece of
    # tag is 3 inches wide
    KNOWN_WIDTH = 1
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    #d:\AI research internship\opencv_scripts\Triangle Similarity
    paths = r'D:\AI research internship\opencv_scripts\Triangle Similarity'
    #image = cv2.imread("images/2ft.png")
    #marker = find_marker(image)
    #focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    #Camera Caliberation 
        



    # Test for a frame Distance  -------------------------------------------------------------------------------------------

    cap = cv2.VideoCapture(0)




    #After running the script Camera_calib.py - the focal lenght I got is: 
    focal_length = int(772.3458251953125) 

    #------------------------------------------------------------------------------------------------------------------------------
    #Start 
    #Track the tag, and calculate the Distance 

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        #frame = cv2.resize(frame, (300, 300))
        
        # Find the marker in the frame
        #marker_a = find_marker_a(frame)
        marker = find_marker_2(frame)
        print('marker is:', marker)
        
        #Find aruco tag
        
        
        if marker is not None:
          if marker > 0:
            # Calculate the focal length using the known distance and width
            
            
            # Calculate the distance in inches
            inches = distance_to_camera(KNOWN_WIDTH, focal_length, marker)

            #brightness

            # Convert 'number' to a formatted string with one decimal place
            formatted_number = "{:.1f}".format(inches)

            bright = isbright(frame)
            
            # Draw a bounding box around the image and display it
            #box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
            #box = np.int0(box)
            #cv2.drawContours(frame, [box], -1, (255, 255, 0), 4)
            
            text = f"Distance: {inches}"

                # Put the text on the right side of the frame
            cv2.putText(frame, text ,
                        (200,200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                    (255, 255, 0), 
                    4, 
                    cv2.LINE_4)



        # Display the frame (whether or not a marker is detected)
        cv2.imshow('img_check', frame)

        # Wait for 1000 milliseconds and check if the user pressed 'x' to exit
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #Calculate ANGLE 
if args.angle:
        # Test for a frame Distance  -------------------------------------------------------------------------------------------
        # Store varibles/load using pickle 
        
        # Given camera matrix
        cameraMatrix = np.array([[643.38354492, 0, 949.42837887],
                                [0, 494.5776062, 519.32249957],
                                [0, 0, 1]])
        #After running the script Camera_calib.py - the focal lenght I got is: 
        focal_length = int(772.3458251953125) 
        k1 = 2.5540818894453188
        k2 = [[9.64358231e+02, 0.00000000e+00, 1.03788434e+03],
            [0.00000000e+00, 9.62492456e+02, 5.88645917e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


        p1 = [[-0.42930121,  0.24517995,  0.00135663, -0.00494753, -0.07929622]]

        p2 =(([[-0.04141919],
            [ 0.18247221],
            [-0.15969552]]), ([[0.06351039],
            [0.0612138 ],
            [0.42058486]]), ([[-0.15129478],
            [ 0.40200938],
            [-0.25175205]]), ([[0.10285869],
            [0.11820688],
            [0.25206441]]), ([[0.10166514],
            [0.13140206],
            [0.25221427]]), ([[0.01618808],
            [0.30460196],
            [0.20265187]]), ([[-0.15396583],
            [ 0.35268659],
            [-0.00901704]]), ([[-0.23392849],
            [-0.25501457],
            [-3.04270686]]), ([[-0.17902949],
            [ 0.36534875],
            [-0.02156775]]), ([[-0.13658845],
            [ 0.38038126],
            [ 0.22615647]]))

        k3 =  (([[-3.45712783],
            [-2.61247496],
            [ 6.92790337]]), ([[ 4.34460648],
            [-4.64416185],
            [10.10966198]]), ([[-6.84016345],
            [-1.30877765],
            [10.58199518]]), ([[ 2.58730669],
            [-3.18810802],
            [ 9.54960441]]), ([[ 3.18078314],
            [-3.59769643],
            [ 9.5760269 ]]), ([[ 1.02130458],
            [-4.53430185],
            [ 9.9197124 ]]), ([[-1.39460395],
            [-2.73733446],
            [ 8.62286697]]), ([[ 4.77999348],
            [ 2.91726858],
            [10.30500898]]), ([[-5.71337612],
            [-1.95494632],
            [ 9.37482064]]), ([[-0.03813453],
            [-4.52980484],
            [ 9.17452092]]))
            
            #distCoeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        k2_flat = np.array(k2).flatten()
        p1_flat = np.array(p1).flatten()
        p2_flat = np.array(p2).flatten()
        k3_flat = np.array(k3).flatten()

        distCoeffs = np.array([k1] + list(k2_flat) + list(p1_flat) + list(p2_flat) + list(k3_flat), dtype=np.float32)

            # Define the marker length in meters
        markerLength = 1

        




        roll_deg = 0
        pitch_deg = 0
        yaw_deg = 0

        def pose_esitmation(frame, matrix_coefficients, distortion_coefficients):

            '''
            frame - Frame from the video stream
            matrix_coefficients - Intrinsic matrix of the calibrated camera
            distortion_coefficients - Distortion coefficients associated with your camera

            return:-
            frame - The frame with the axis drawn on it
            '''


        

            ids, corners = find_marker_2(frame)

                #Export values fro Camera Calib:

            

                # Load imgpoints and objpoints using pickle
            with open('imgpoints.pkl', 'rb') as f:
                    imgpoints= pickle.load(f)
                    print('imgpoints: ', imgpoints)

            with open('objpoints.pkl', 'rb') as f:
                    objpoints = pickle.load(f)
                    print('objpoints: ', objpoints)
                    objpoints = np.random.random((10,3,1))
                    imgpoints = np.random.random((10,2,1))
                    matrix_coefficients = np.eye(3)

                    distortion_coefficients = np.zeros((5,1))
                    

                    # If markers are detected
            if len(corners) > 0:
                        # Convert the list of arrays to a numpy array
                        
                    #for i in range(0, len(ids)):
                        # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                        _, tvec, rvec = cv2.solvePnP(objpoints,imgpoints, matrix_coefficients,
                                                                                distortion_coefficients)
                        # Draw a square around the markers
                        cv2.aruco.drawDetectedMarkers(frame, corners) 

                        #rvecs, tvecs, _ = cv2.aruco.solvePnP(np.array(corners), markerLength, cameraMatrix, distCoeffs)
                        print('rvec is', rvec)
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

                        # Convert angles from radians to degrees
                        roll_deg = np.degrees(roll)
                        pitch_deg = np.degrees(pitch)
                        yaw_deg = np.degrees(yaw)

                        text2 = f"Roll: {roll_deg}, Pitch: {pitch_deg}, Yaw: {yaw_deg}"

                        frame = cv2.drawFrameAxes( frame, matrix_coefficients, distortion_coefficients, rvec, tvec, length=0.003 )

                        cv2.putText(frame, text2,
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1,  # Font scale
                                    (255, 255, 0),  # Text color (in BGR)
                                    2,  # Line thickness
                                    cv2.LINE_AA)  # Line type (anti-aliased)
                        print('opencv version', cv2.__version__)
                            # Draw Axis
                        
                        #cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  



            return frame

    # Assuming you have defined KNOWN_WIDTH, focal_length, cameraMatrix, distCoeffs, and other functions

        cap = cv2.VideoCapture(0)  # Open the video stream


        while True:
            ret, frame = cap.read()

            marker, corners= find_marker_2(frame)
            print('OBJECT POINTS IS : ',corners)
            print('marker is:', marker)

            output = pose_esitmation(frame, cameraMatrix, distCoeffs)

            cv2.imshow('Estimated Pose', output)
                
            # Wait for 1 millisecond and check if the user pressed 'x' to exit
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()
# -----------------------------------------------------------------------------------------------------
if args.empty:

    #if string contains empty add set to an array 
    empty_dir = r'D:\AI research internship\opencv_scripts\empty_data2'
    #img_dict = load_images(path)
  
    def original_id_empty(image_dict):
        arr = []
        for key in image_dict.keys():
            if 'empty' in key:
                empty_set = set()
                arr.append(empty_set)
        print(arr)
        return arr

    #original_id_empty(img_dict)
    def info_empty_TN(TP, TN, FP, FN, FP2, scores, total):
            precision = 0
            recall = 0

            if TP + FP + FP2 != 0:
                precision = TP / (TP + FP + FP2)
            else:
                precision = 0

            if TP + FN != 0:
                recall = TP / (TP + FN)
            else:
                recall = 0

            print(f'Predicted: Score:{(scores/total)*100}% | TP:{TP}, FN:{FN}, TN:{TN}, FP1:{FP}, FP2:{FP2} | precision:{precision}, recall:{recall}')
            print(f'Out of {total} images, {TN} were predicted accurately')

    def pplns_batch_empty(ppln, directory):
            batch_size = 150
            for batch in load_images_in_batches(directory, batch_size):
                original_ids = original_id_empty(batch)
                #print(original_ids)
                predicted_ids = ppln(batch)
                #cvs(original_ids,predicted_ids, batch)
                #new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
                #print(predicted_ids)
                TP, TN, FP, FN, FP2, scores, total = calc2(original_ids,predicted_ids)
                info_empty_TN(TP, TN, FP, FN, FP2, scores, total)

    pplns_batch_empty(F_2_ppln, empty_dir)

#Test adding features as angle and distance 
string = 'a_5_d_20'
def get_angle_distance(string):
    result_dict = {}
    
    parts = string.split('_')
    for i in range(len(parts) - 1):
        if parts[i] == 'a':
            result_dict['Angle'] = int(parts[i + 1])
        elif parts[i] == 'd':
            result_dict['Distance'] = int(parts[i + 1])
    
    return result_dict


 
     
#df_distance_angle = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])

def cvs(original_ids, predicted_ids,  batch):
        # Create a DataFrame with the 'IDs', 'Predictions', and 'Batch' columns
        df = pd.DataFrame(columns=['IDs', 'Predictions', 'Batch'])

        # Convert empty sets to None
        original_ids_1, predicted_ids_1  = convert(original_ids,predicted_ids)

        #classification arr

        classification = class_arr(original_ids,predicted_ids)

        # Create DataFrame from the second function's output
        df_classification = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])
        # Add Angle and Distance columns using the get_angle_distance function
        df_classification['Angle'] = get_angle_distance('a_5_d_20')['Angle']
        df_classification['Distance'] = get_angle_distance('a_5_d_20')['Distance']
        # Fill the DataFrame with data from original_ids and predicted_ids
        batch_n = 1
        for original_id, predicted_id in zip(original_ids_1, predicted_ids_1):
            df = pd.concat([df, pd.DataFrame({'IDs': [original_id], 'Predictions': [predicted_id], 'Batch': [batch_n]})], ignore_index=True)
            batch_n+=1
        # Concatenate both DataFrames side by side
        result_df = pd.concat([df, df_classification], axis=1)
        result_df.to_csv('final_dataframe.csv', index=False)

# if a angle:, next value put to the key, if d - distance: , next value put to the key 
# and add do cvs file as separate 2 columns 
'''
#Test running F ppln 
cap = cv2.VideoCapture(0)  # Open the video stream


while True:
    ret, frame = cap.read()

    marker, corners= find_marker_2(frame)
    print('OBJECT POINTS IS : ',corners)
    print('marker is:', marker)

    output = pose_esitmation(frame, cameraMatrix, distCoeffs)

    cv2.imshow('Estimated Pose', output)
        
    # Wait for 1 millisecond and check if the user pressed 'x' to exit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
'''


#----------------Calculate Angle ---------------------------------------------------------------------------------------
 # Calculates angle based off the difference between theoretical width and
    # actual width
#Camera Calib + saving pictures witht the patterns 
#Reference : https://longervision.github.io/2017/03/16/ComputerVision/OpenCV/opencv-internal-calibration-chessboard/




# corners is a list of arrays, where each array contains the corner points of a detected marker
# You need to populate corners based on your marker detection result
  # Fill in the detected corner points

# Call the estimatePoseSingleMarkers function

  
#https://docs.opencv.org/4.x/d9/d6a/group__aruco.html






# -------------------------------------USEFULL LINKS & RESOURCES  
#UI and distence/Angle - https://medium.com/analytics-vidhya/how-to-track-distance-and-angle-of-an-object-using-opencv-1f1966d418b4
#Github Code: https://github.com/ariwasch/OpenCV-Distance-Angle-Demo/blob/master/Vision.py#L115

#Color pale2tte - https://rgb.to/hex/ffff00 

#LIDAR - https://www.thinkautonomous.ai/blog/how-lidar-detection-works/


# 3D - reconstraction https://medium.com/vitalify-asia/create-3d-model-from-a-single-2d-image-in-pytorch-917aca00bb07

#CARLA - http://carla.org/

#Generete 3d models using images HomeArtificial IntelligenceHow to generate 3D Models using Images | Machine Learning.
#How to generate 3D Models using Images | Machine Learning.

#Keywords : map image to a 3d model, onvolutional AutoEncoder vs cnn, Lidar Detection (Light detection),
# craeat 3d model from images from cnn

#Learning a Probabilistic Latent Space 
# of Object Shapes via 3D Generative-Adversarial Modeling(3D GAN)
#Possibly collect data from sensors 