# Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-
Developing scoring metrics for the image processing pipeline 

Step # 0 What is the project about? Gain an understanding of what is it, how it works, and it's  goal. 

Step # 1 Identifying what do I want to work on -  Focuse on the Computer Vision part of the project. I build an image processing pipeline, which detects arUco tag. It is important because it maps deteted ids of a physcial piece of hardware to a dgital version (CAD model)

Step #2 I am working on scoring metrics, which help to optimize the image processing pipeline perfomence. 

Step #3 Gathering Data for the Scoring system. 
 + first attempt - I collected 30 images of arUco tags (it took me 4hr)
 + secod attempt - I collected 7K images of arUco tags, but this time I tried to capture angle: From left to right, and distance : up/down shots. (I didn't really measure an accuracy, and it was my first attempt of trying to capture a first condition, whcih would be a good start for a first detailed feedback of a scoring fucntion.)
 + third attempt - I collected 5K data and this time I manually measured an angle, distance and described the ligitng condition (10 various scenarios).
 + Example ->   data_1: 'Angle: 70, Distance: 4, Lighting: low day light at 5:48 PM',

