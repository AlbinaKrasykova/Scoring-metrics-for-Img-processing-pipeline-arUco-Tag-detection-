**Scoring Metrics for Image Processing Pipeline: arUco Tag Detection**

![image](https://github.com/AlbinaKrasykova/Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-/assets/91033995/5f1421e4-863f-4e5f-b3e3-5f454d4428f0)


API 

Step 1
```
import math
import os
import re
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)
```

Step 2
```
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
```

Tips (from what have I learned):

1. Break down the problem in simple steps
2. Find the simplest possible solution
3. Whether you debug an application or try to build a solution . Start SIMPLE !()
Example:
4. Load the data in batches - it allows us to process data quickly.
5. Try to keep code organized from the beginning, document buggs, challenges, solutions.
6. Upload updated code, and metadata to the github. 
