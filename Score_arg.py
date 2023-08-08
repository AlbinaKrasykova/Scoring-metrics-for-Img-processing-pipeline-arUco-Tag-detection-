# importing librraies 

import argparse 
from shadow_highlight_correction import correction
import math
import os
import re
from PIL import Image
import pandas as pd 
import imutils
import numpy as np
import cv2 as cv
import cv2

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


#0. Function that Generates dictionary with images from the file - ✔
# Key: 9_angle_6_.png, Value: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x178C0FF6BF0>



#1 Loads images in batches 

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

# F_ppln CREATE an ARRAY with Tags that were predicted - ✔

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

def pplns(ppln):
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

#6 Calculates the precison and recall - ✔ 
#if original is empty set(), and predicted is empty TN++  EX:  (set() - > set()) - TN++
#id the original is empty set(, and predicted is something else then empty FP++ EX: (set() -> {17}) - FP++ 
#ALso if the set is one value, and th epredicted is anothe value, FP++ EX: ({23,40} -> {17}) - FP++ 


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

#TO ADD: toal FP/TN/FN/TP
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




cap = cv2.VideoCapture(0)


#Command_Line -----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--my_score', action='store_true')
parser.add_argument('--Fahad_score', action='store_true')
parser.add_argument('--Amrit_score', action='store_true')
parser.add_argument('--score_2id', action='store_true')
parser.add_argument('--score_all_cases', action='store_true')

args = parser.parse_args()

output_filename = 'output.doc'  
# Save the output image to a file


if args.my_score:

    batch_size = 150
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'


    def My_ppln(image_dict):
        p_id_arr = []
        for ids, img in image_dict.items():
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clache = cv2.createCLAHE(clipLimit=40)
            frame_clache = clache.apply(gray)           
            th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 1)
            blurred = cv2.GaussianBlur(th3, (21, 21), 0)
            #flipped = cv2.flip(blurred, 1)
            _, ids, _ = detector.detectMarkers(blurred)
            print('id dtype is ', type(ids))
            p_id_arr.append(ids)

        print('My array with predicted id is: ', p_id_arr)
        return p_id_arr
    
    
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
        

    #small dataset 41 img - 1id 
    directory0 = r'D:\AI research internship\opencv_scripts\data_set'
    img_dict = load_images(directory) 
 


   # raturns values, and array with the sets of classification
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
    import pandas as pd
    import os

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

        #print(result_df)

        file_path = 'data999-final0.csv'
        #uncommednt once create new datset
        #result_df.to_csv(file_path, mode='a', index=False, header=not os.path.isfile(file_path))
        #print('Successfully saved to the file!')



 
 # TASK: add original, predicted , classification, batch_number to the dicitonary - > write dictionary to a cvs file  
    def pplns(ppln):
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


#Dataset

    orig_set = original_id_2(img_dict)

    predict_set = F_2_ppln(img_dict)

    
    total_TP, total_FP, total_FN, true_negative, false_positive2, scores, total = fixed_calc_p_r(orig_set, predict_set)

    
       

    a_orig_set = [set(),{40,30},{5},{40,30},{6},{20,30},set()]

    a_predicted_set = [set(),{17},{5},{40,30},set(),{20},{5}]

    total_TP_a, total_FP_a, total_FN_a, true_negative_a, false_positive2_a, scores_a, total = fixed_calc_p_r(a_orig_set, a_predicted_set)

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

#TEST  ----------------------------------------------------------------------------------------------- FINDING DISTANCE

#Test data Triangle Similarity 
#from imutils import paths





def find_marker_a(img):
          
    transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transformation = cv2.bitwise_not(transformation)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
    transformation = clahe.apply(transformation)

    transformation = cv2.GaussianBlur(transformation, (21, 21), 0)

    transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)

    _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)  # Renamed 'transformation' to 'de
    corners,ids,_ = detector.detectMarkers(transformation)   
    detected_markers = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    return detected_markers 

                        

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


# initialize the known distance from the camera to the object, which
# in this case is 11 cm
KNOWN_DISTANCE = 11
# initialize the known object width, which in this case, the piece of
# tag is 3 inches wide
KNOWN_WIDTH = 3
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
#d:\AI research internship\opencv_scripts\Triangle Similarity
paths = r'D:\AI research internship\opencv_scripts\Triangle Similarity'
#image = cv2.imread("images/2ft.png")
#marker = find_marker(image)
#focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#Camera Caliberation 




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
    
    return marker



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
	



# Test for a frame Distance  -------------------------------------------------------------------------------------------

cap = cv2.VideoCapture(0)
import cv2 

IMAGE_PATHS = ['(1).jpeg']
KNOWN_DISTANCE = 4.3
KNOWN_WIDTH = 1.2

image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
#focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#CAMERA CALIB  -------------------------------------------------------------------------------------------------------------
directory = "D:/AI research internship/opencv_scripts/checkboard_data"
image_files = os.listdir(directory)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



count = 0
for image_file in image_files:
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(directory, image_file)
        img = cv2.imread(image_path)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
        
        if ret:
            print(f"Corners found in image: {image_file}")
            
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Visualize detected corners
            img_with_corners = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
            cv2.imshow("Corners", img_with_corners)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        else:
            print(f"No corners found in image: {image_file}")
            
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# Close all windows at the end
cv.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------------------
#Start 
#Track the tag, and calculate the Distance 

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    #frame = cv2.resize(frame, (300, 300))
    
    # Find the marker in the frame
    marker = find_marker(frame)
    
    if marker is not None:
        # Calculate the focal length using the known distance and width
        
        
        # Calculate the distance in inches
        inches = distance_to_camera(KNOWN_WIDTH, focal_lenght, marker[1][0])

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