# AI internship Part - 1 

#Step #1  - importing libraries  

import math
import os
import re
import cv2
import pandas as pd 


# Step 2 collecting metadata of aruco tags, capturing various conditons as angle, light, distance 
# Goal: helps to see how well or bad ppl perfomed under different conditions clearly  
# LOAD the DATA (10 datasets with different condiitons as : light, angle, distance)




dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)

def clean_string(string):
        digits = re.findall(r'\d+', string)
        if len(digits) > 1:
            digits = digits[:-1]  
        cleaned_string = '_'.join(digits)
        count = len(digits)
        
        return count, cleaned_string




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


def get_original_id(image_dict):
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

    
def ppln_transformations(ids,img):

 transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 transformation = cv2.bitwise_not(transformation)
 clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
 transformation = clahe.apply(transformation)
 transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
 transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)
 _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY) 
 _,ids,_ = detector.detectMarkers(transformation)  
 return ids
  
  # before called - img_p_ppln
def get_predicted_id(image_dict,ppln_transformations):
            arr = []
            
            p_id_set = set()
            
            for ids, img in image_dict.items():
                ids = ppln_transformations(ids,img)
                
                   
                p_id_set = set()
                
                if ids is not None:
                    
                    for inner_arr in ids:
                        for i in inner_arr:
                            p_id_set.add(i)
                        
                arr.append(p_id_set)
            print('F_2 predictions: ')    
            print(arr) 
            return arr


def cvs_all(original_ids, predicted_ids,  folder_name):
    # Create a DataFrame with the 'IDs', 'Predictions', and 'Batch' columns
    df = pd.DataFrame(columns=['IDs', 'Predictions', 'Folder', 'Angle', 'Distance', 'Light'])

    # Convert empty sets to None
    original_ids_1, predicted_ids_1  = convert(original_ids, predicted_ids)

    # Classification arr
    classification = class_arr(original_ids, predicted_ids)

    # Create DataFrame from the second function's output
    df_classification = pd.DataFrame(classification, columns=['TP', 'FN', 'FP', 'TN'])
    
    # Add Angle and Distance columns using the get_angle_distance function
    # Get the name 
    angle, distance  = get_angle_distance(folder_name)
    
    # Create a list to store Angle and Distance values
    angle_list = [angle] * len(original_ids_1)
    distance_list = [distance] * len(original_ids_1)
    
    # Fill the DataFrame with data from original_ids and predicted_ids
    for original_id, predicted_id in zip(original_ids_1, predicted_ids_1):
        df = pd.concat([df, pd.DataFrame({'IDs': [original_id], 'Predictions': [predicted_id], 'Folder': [folder_name]})], ignore_index=True)

    # Assign the Angle and Distance values to the DataFrame outside the loop
    df['Angle'] = angle_list
    df['Distance'] = distance_list

    # Concatenate both DataFrames side by side
    result_df = pd.concat([df, df_classification], axis=1)
    print('RESULT DF', result_df)
    return result_df



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
    #return array with classification TP,FP,FN,TN [[0,1,0,0],[1,0,0,0]]
    #total_TP, total_FP, total_FN, true_negative, false_positive2, scores, total
    #TO DO: recheck if it produces correct output 
    
        
    

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





def classify_rates_for_cvs(original_ids, predicted_ids):
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
    

def save_df(folder_name, original_ids,predicted_ids):
    df = cvs_all(original_ids, predicted_ids, str(folder_name))
    df.to_csv(f'folder{folder_name}.csv')
    print('DF SAVED!!!')
      
def scores(ppln_transformations, directory, folder_name, batch_size=250):
    batch_size = 250
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = get_original_id(batch)
        predicted_ids =  get_predicted_id(batch,ppln_transformations)               
        
        save_df(folder_name, original_ids,predicted_ids)               
        TP, TN, FP, FN, FP2, scores, total = classify_rates_for_cvs(original_ids,predicted_ids)
        info(TP, TN, FP, FN, FP2, scores, total)

def get_angle_distance(string):
    split_array = string.split('_')
    angle = split_array[1]
    distance = split_array[3]
                    
    return angle, distance 

def score(path, name, ppln):
     
    for key, value in path.items():
        folder_name = name[key]

        print("-----------------------------------------------------------------------------")
        print(f'Folder: {folder_name}')
        print()
        print('Conditions:', value)
        print()
        scores(ppln, key, str(folder_name))
        save_df
        print("-----------------------------------------------------------------------------")


# -----------------------------------------------------------------------------------------------------------------------------
#Implementation 
  # 12 - "D:\AI research internship\opencv_scripts\empty_data2"

data_1 =  r'D:\AI research internship\opencv_scripts\a_5_d_20'
data_2 = r'D:\AI research internship\opencv_scripts\a_28_d_4'
data_3 = r'D:\AI research internship\opencv_scripts\a_70_d_10'
        

name = {
    data_1: 'a_5_d_20',
    data_2: 'a_28_d_4',
    data_3: 'a_70_d_10'
}

path = {
    data_1: 'Angle: 5, Distance: 20, Lighting: Bright',
    data_2: 'Angle: 28, Distance: 4, Lighting: Bright',
    data_3: 'Angle: 70, Distance: 10, Lighting: Dark'
}


def ppln_transformations(ids,img):

 transformation = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 transformation = cv2.bitwise_not(transformation)
 clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
 transformation = clahe.apply(transformation)
 transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
 transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)
 _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY) 
 _,ids,_ = detector.detectMarkers(transformation)  
 return ids






# All you need is
# 1- define variables with the path (depands on have many folders with the metadata you have)
# 2 defined dicitonary 'name' with the name of variables from the # 1 
# 3 define dictionary with the description of each folder with the metadata 
# 4 Call the function score and pass parameters:   path (dtype: dictionary), name(dtype: dictionary) 
# and the pipelien transformation(see the example).  
#Output: printed score, and saved cvs files with the data 


score(path, name, ppln_transformations)
