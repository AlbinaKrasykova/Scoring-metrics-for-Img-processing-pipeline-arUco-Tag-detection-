**Scoring Metrics for Image Processing Pipeline: arUco Tag Detection**

![image](https://github.com/AlbinaKrasykova/Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-/assets/91033995/5f1421e4-863f-4e5f-b3e3-5f454d4428f0)


API --------------------------------------------------------------





# Simple Implementation 

# All you need is

#Output: printed score, and saved cvs files with the data 

# 1 
define variables with the path (depands on have many folders with the metadata you have)

```

data_1 =  r'D:\AI research internship\opencv_scripts\a_5_d_20'
data_2 = r'D:\AI research internship\opencv_scripts\a_28_d_4'
data_3 = r'D:\AI research internship\opencv_scripts\a_70_d_10'
        
```

# 2 
defined dicitonary 'name' with the name of variables from the # 1 
```
name = {
    data_1: 'a_5_d_20',
    data_2: 'a_28_d_4',
    data_3: 'a_70_d_10'
}
```

# 3 
define dictionary with the description of each folder with the metadata 
```
path = {
    data_1: 'Angle: 5, Distance: 20, Lighting: Bright',
    data_2: 'Angle: 28, Distance: 4, Lighting: Bright',
    data_3: 'Angle: 70, Distance: 10, Lighting: Dark'
}


```
# 4
Define the pipelien transformation(see the example).  

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
Call the function score and pass parameters:   path (dtype: dictionary), name(dtype: dictionary) 

```
score(path, name, ppln_transformations)

```

Tips (from what have I learned):

1. Break down the problem in simple steps
2. Find the simplest possible solution
3. Whether you debug an application or try to build a solution . Start SIMPLE !()
Example:
4. Load the data in batches - it allows us to process data quickly.
5. Try to keep code organized from the beginning, document buggs, challenges, solutions.
6. Upload updated code, and metadata to the github. 
