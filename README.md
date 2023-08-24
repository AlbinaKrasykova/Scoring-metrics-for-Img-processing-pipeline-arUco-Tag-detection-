**Scoring Metrics for Image Processing Pipeline: arUco Tag Detection**

Developing scoring metrics for the image processing pipeline.

**Step 0:** **What is the project about?** Gain an understanding of what it is, how it works, and its goal.

**Step 1:** **Identifying what I want to work on** - Focus on the Computer Vision part of the project. I am building an image processing pipeline that detects arUco tags. This is important because it maps detected IDs of a physical piece of hardware to a digital version (CAD model).

**Step 2:** **Working on scoring metrics**, which help optimize the image processing pipeline performance.

**Step 3:** **Gathering Data for the Scoring System.**
- **First attempt:** I collected 30 images of arUco tags (it took me 4 hours).
- **Second attempt:** I collected 7,000 images of arUco tags, but this time I tried to capture different angles: from left to right, and distances: up/down shots. (I didn't measure accuracy, and it was my first attempt to capture a condition that would be a good start for detailed feedback on a scoring function.)
- **Third attempt:** I collected 5,000 data points, and this time I manually measured angles, distances, and described the lighting conditions (10 various scenarios).
- **Example:** data_1: 'Angle: 70, Distance: 4, Lighting: Low daylight at 5:48 PM',
