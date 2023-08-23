# Scoring-metrics-for-Img-processing-pipeline-arUco-Tag-detection-
Developing scoring metrics for the image processing pipeline 

Step # 0 What is the project about? Gain an understanding of what is it, how it works, and it's  goal. 

Step # 1 Identifying what do I want to work on -  Focuse on the Computer Vision part of the project. I build an image processing pipeline, which detects arUco tag. It is important because it maps deteted ids of a physcial piece of hardware to a dgital version (CAD model)

Step #2 I am working on scoring metrics, which help to optimize the image processing pipeline perfomence. 

Step #3 Gathering Data for the Scoring system. 
 + first attempt - I collected 30 images of arUco tags (it took me 4hr)
 + secod attempt - I collected 7K images of arUco tags, but this time I tried to capture angle: From left to right, and distance : up/down shots. (I didn't really measure an accuracy, and it was my first attempt of trying to capture a first condition, whcih would be a good start for a first detailed feedback of a scoring fucntion.)
 + third attempt - I collected 5K data and this time I manually measured an angle, distance and described the ligitng condition.
          data_1: 'Angle: 70, Distance: 4, Lighting: low day light at 5:48 PM',
        data_2 : 'Angle: 70, Distance: 10, Lighting: low day light at 7:50 PM',
        data_3: 'Angle: 28, Distance: 10, Lighting: low day light at 8:48 PM',
        data_4: 'Angle: 5, Distance: 4, Lighting: low day light at 3:58 PM',
        data_5: 'Angle: 50, Distance: 10, Lighting: bright artificial and day light at 7:53 ',
        data_6: 'Angle: 50, Distance: 4, Lighting: bright artificial light low evening light at 7:01 PM',
        data_7: 'Angle: 70, Distance: 20, Lighting: low day light at 7:41 PM',
        data_8: 'Angle: 50, Distance: 20, Lighting: bright artificial light at 9:32 PM',
        data_9 : 'Angle: 28, Distance: 20, Lighting: bright artificial light at 9:49 PM',
        data_10: 'Angle: 5, Distance: 20, Lighting: bright artificial light at 10:08 PM'
 + a better way- ? Live tracking of various conditions 
