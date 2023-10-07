                                                #  **Scoring Metrics for Image Processing Pipeline: arUco Tag Detection**

  1. API explained
  2. Collecting and labeling the metadata
  3. EDA
  4. What I learned summary 



![image](https://github.com/AlbinaKrasykova/Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-/blob/main/image%20(3).png)


API --------------------------------------------------------------





# API & Detailed explanation  


#0 Download the folders with the metadata from the repo.
This reposatory has 3 folders. Each folder represents a condition. For example folder - 'a_5_d_20_B' stands for (a)aangle: 5, (d)distance: 20, (b) - bright 700Lux lighting.  

![image](https://github.com/AlbinaKrasykova/Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-/blob/main/image%20(5).png)

# 1 
define variables, string format  with the path of the folder. Where data_1 variable refers to a folder with a condition. 

```

data_1 =  r'D:\AI research internship\opencv_scripts\a_5_d_20'
data_2 = r'D:\AI research internship\opencv_scripts\a_28_d_4'
data_3 = r'D:\AI research internship\opencv_scripts\a_70_d_10'
        
```

# 2 
defined dicitonary, let's call it - 'name' with the name of the folders(that contains metadata) from the  step # 1 
```
name = {
    data_1: 'a_5_d_20',
    data_2: 'a_28_d_4',
    data_3: 'a_70_d_10'
}
```

# 3 
define dictionary with the description of each folder. Exampe: key - data_1(initialize the folder with the metadata_1), value contains a string with a description of a condition. 
```
path = {
    data_1: 'Angle: 5, Distance: 20, Lighting: Bright',
    data_2: 'Angle: 28, Distance: 4, Lighting: Bright',
    data_3: 'Angle: 70, Distance: 10, Lighting: Dark'
}


```
# 4
Define the function with the image processing pipeline, that contains image preprocessing steps and detecting returning ids only .  

```
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

```



# 4 
Finally call the function score and pass the parameters: from the step #3 -  path (dtype: dictionary), from the step #2 name (dtype: dictionary) 

```
score(path, name, ppln_transformations)

```
# 5

Output: printed score, and saved cvs files with the data 

![image](https://github.com/AlbinaKrasykova/Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-/blob/main/score.png)

Tips (from what have I learned):

1. Break down the problem in simple steps
2. Find the simplest possible solution
3. Whether you debug an application or try to build a solution . Start SIMPLE !()
Example:
4. Load the data in batches - it allows us to process data quickly.
5. Try to keep code organized from the beginning, document buggs, challenges, solutions.
6. Upload updated code, and metadata to the github. 
