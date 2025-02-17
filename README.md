# Scene-Synthesis-using-SFM

# Introduction
In this project, we reconstructed a 3D scene and simultaneously obtained the camera poses with respect to the scene, with a given set of 6 images from a monocular camera and their feature point correspondences. Following are the steps involved:

Feature detection and finding correspondences
Estimating Fundamental Matrix
Essential Matrix and solving for camera poses
Linear Triangulation and recovering correct pose
Non Linear Triangulation
Linear PnP, RANSAC and Non linear optimization
Bundle Adjustment

# Pipeline
![alt text](https://github.com/DhirajRouniyar/Scene-Synthesis-using-SFM/blob/main/Images/pipeline.png)

# Results
![alt text](https://github.com/DhirajRouniyar/Scene-Synthesis-using-SFM/blob/main/Images/Results.png)

# How to run the code
Run Wrapper.py 

Before that 
1. Make Data folder outside Phase 1 folder and save images and matching poitns given to us
2. Make outputs folder outside Phase 1
   Inside outputs make folder to save the .npy files
              ---BA
              ---No_BA
          


To see Graphs:

3. Run See_save_plot.py

4. In See_save_plot.py uncomment line 20,21,22,23 and comment line 30,31,32,33 to get plot for triangulation vs Nontriangulation points
