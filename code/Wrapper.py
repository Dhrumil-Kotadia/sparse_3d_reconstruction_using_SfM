import numpy as np
import cv2

from Load_Data import getMatchingFeatures
# from EstimateFundamentalMatrix import *   #EstimateFundamentalMatrix ia used in RANSAC_GetInliers.py
from GetInliersRANSAC import getInliers
from EssentialMatrixfromFundamentalMatrix import getEssentialMatrixfromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from NonLinearTriangulation import NonLinearTriangulation, ReprojectionLoss 
from DisambiguateCameraPose  import DisambiguatePose, DepthPositivityConstraint
from LinearPnP import reprojectionErrorPnP, PnP
from PnPRansac import PnPError, PnPRANSAC
from NonLinearPnP import NonLinearPnP, PnPLoss
from BundleAdjustment import BundleAdjustment
from Utils.ImageUtils import readImageSet, makeImageSizeSame, showMatches, showAllMatches, showMatches_filtered, showFilteredMatches
from Utils.MiscUtils import foldercheck, project3DPoints, ProjectionMatrix, meanReprojectionError, ReprojectionError, projectPts, homography, getQuaternion, getEuler, getRotation 
from matplotlib import pyplot as plt
import argparse
# import See_save_plot
import csv


# Number of IMAGES given
# total_images = 5

# Camera Intrinsic parameters:
K = np.array([[531.122155322710,             0,  407.192550839899],
              [            0, 531.541737503901, 313.308715048366],
              [            0,             0,             1]]).reshape(3,3)

# Writing main based on Pipeline given
def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--LoadDataPath', default="../Data/", help='base path for data')
    Parser.add_argument('--SavePath', default="../outputs/", help='Save file')
    
    Args = Parser.parse_args()
    folderpath = Args.LoadDataPath
    savepath = Args.SavePath
    
    #for error chart
    f = open('error_chart.csv', mode='w')
    error_chart = csv.writer(f)
    ##
    total_images = 5
    images = readImageSet(folderpath, total_images)
    ### Obtain Feature point, Extract Inlier points by RNASAC, Calculate Fundamental matrix, and Filter feature flags for Inlier correspondance points ###
    """
    Extract Feature points: 
    -From given matching files between images list down all x and y correspondance points and
    -generate feature_flag where the non zero column positions signify that there are point correspondences between those images
    -feature flag columns are images and rows are feature points
    -we later generate id whixh signifies feature flag one image to other image correspondance ie. feature flag btwn one image to other is 1.
    -feature_descriptor are just RGB values of the feature points - no use in this pipeline
    """

    feature_x, feature_y,  feature_flag, feature_descriptor = getMatchingFeatures(folderpath, total_images)
    
    # feature_flag_filtered empty array
    feature_flag_filtered = np.zeros_like(feature_flag)

    # f_matrix stores the fundamental matrix between images
    f_matrix = np.empty(shape=(total_images, total_images), dtype=object)

    for i in range(0, total_images - 1):
        
        for j in range(i + 1, total_images):

            id = np.where(feature_flag[:,i] & feature_flag[:,j])  # id of feature points that are common between i and j
            pts1 = np.hstack((feature_x[id, i].reshape((-1, 1)), feature_y[id, i].reshape((-1, 1)))) # feature points in image i
            pts2 = np.hstack((feature_x[id, j].reshape((-1, 1)), feature_y[id, j].reshape((-1, 1)))) # feature points in image j
            # reshape id to 1D array
            id = np.array(id).reshape(-1)
            
            if len(id) > 8:
                F_for_Inliners, id_for_Inliners = getInliers(pts1, pts2, id)
                print( 'Num of inliers: ', len(id_for_Inliners), '/', len(id), '_At image : ',  i,j, )            
                f_matrix[i, j] = F_for_Inliners
                feature_flag_filtered[id_for_Inliners, j] = 1
                feature_flag_filtered[id_for_Inliners, i] = 1
    # showFilteredMatches(images, feature_x, feature_y, feature_flag_filtered, feature_flag, total_images)
    # np.save('./tmp_files/WPI/feature_flag_filtered.npy',feature_flag_filtered)
    # np.save('./tmp_files/WPI/f_matrix.npy',f_matrix)  
    # print("Fmatrix", f_matrix)              
    

    ### Calculate Essential Matrix from Fundamental matrix, Estimate Camera Pose R and C, Use Triangulation to get 3D point ###
    # First lets register two Images 1 and 2
    g,h = 0,1 
    print('Image 1 and Image 2 is registering...')
    
    # Fundamental matrix between image 1 and 2
    F12 = f_matrix[g,h]
    print('Fundamental Matrix between Image 1 and Image 2: ', F12)

    # Essential matrix between image 1 and 2
    E12 = getEssentialMatrixfromFundamentalMatrix(K, F12)
    print('Essential Matrix between Image 1 and Image 2: ', E12)

    # Extracting 2nd Camera Pose R and C from Essential matrix
    Rot_2, C_2 = ExtractCameraPose(E12)
    print('Camera 2 Poses: ', Rot_2, C_2)

    # Extracting feature points that are common between image 1 and 2
    id = np.where(feature_flag_filtered[:,g] & feature_flag_filtered[:,h])
    pts1 = np.hstack((feature_x[id, g].reshape((-1, 1)), feature_y[id, g].reshape((-1, 1))))
    pts2 = np.hstack((feature_x[id, h].reshape((-1, 1)), feature_y[id, h].reshape((-1, 1))))
    
    # First Camera Pose let's assume it to be at origin
    Rot_1 = np.identity(3)
    C_1 = np.zeros((3,1))
    I = np.identity(3)

    # Extract 3D points using Linear Triangulation
    _3D_pts = []
    for i in range(len(C_2)):
    
        x1 = pts1
        x2 = pts2
        X = LinearTriangulation(K, C_1, Rot_1, C_2[i], Rot_2[i], x1, x2)
        X = X/X[:,3].reshape(-1,1)
        _3D_pts.append(X)
    
    # Extracting the best 3D points using DisambiguatePose - chiralaty check
    Selctd_Rot_2, Selctd_C_2, X_3d_Chirality_OK = DisambiguatePose(Rot_2, C_2, _3D_pts)
    X_3d_Chirality_OK = X_3d_Chirality_OK/X_3d_Chirality_OK[:,3].reshape(-1,1)
    print('### Selected the Camera pose which satifies Chirality after Linear Triangulation ###')

    # Get Optimized 3D points using NonLinear Triangulation - optimize by least square for Reprojection Loss
    print('### Calculating NonLinear Triangulation ###')
    X_3d_optimized = NonLinearTriangulation(K, pts1, pts2, X_3d_Chirality_OK, Rot_1, C_1, Selctd_Rot_2, Selctd_C_2)
    X_3d_optimized = X_3d_optimized / X_3d_optimized[:,3].reshape(-1,1)
    
    # Mean Reprojection Error before and after optimization (refining 3D points)
    mean_error_lt = meanReprojectionError(X_3d_Chirality_OK, pts1, pts2, Rot_1, C_1, Selctd_Rot_2, Selctd_C_2, K )
    mean_error_nlt = meanReprojectionError(X_3d_optimized, pts1, pts2, Rot_1, C_1, Selctd_Rot_2, Selctd_C_2, K )
    print(g+1,h+1, 'After Lt and before optimization nLT: ', mean_error_lt, 'After optimization nLT:', mean_error_nlt)
    
    print("mean_error_lt", mean_error_lt)
    print("mean_error_nlt", mean_error_nlt)
    error_row_for_chart = np.zeros((20))
    error_row_for_chart[3] = mean_error_lt
    error_row_for_chart[9] = mean_error_nlt
    error_chart.writerow(list(error_row_for_chart))

    ## Register Camera 1 and 2  ##

    X_3d_all = np.zeros((feature_x.shape[0], 3))
    camera_indices = np.zeros((feature_x.shape[0], 1), dtype = int) 
    X_found_flag = np.zeros((feature_x.shape[0], 1), dtype = int)

    X_3d_all[id] = X_3d_Chirality_OK[:, :3]
    X_found_flag[id] = 1
    camera_indices[id] = 1

    X_found_flag[np.where(X_3d_all[:,2] < 0)] = 0

    C_reg_set_ = []
    R_reg_set_ = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_reg_set_.append(C0)
    R_reg_set_.append(R0)

    C_reg_set_.append(Selctd_C_2)
    R_reg_set_.append(Selctd_Rot_2)
    print(' #####################  Done registration Cameras 1 and 2 #####################' )
    
    
    ### Register Remaining Cameras ### 
    print('Remaining Images are registering......')
    total_images = 3
    for i in range(2, total_images):
        # for chart
        error_row_for_chart = np.zeros((20))
        #for chart

        print('Registering Image: ', str(i+1) ,'......')
        id_feature_i = np.where(X_found_flag[:, 0] & feature_flag_filtered[:, i])
        if len(id_feature_i[0]) < 8:
            print("Found ", len(id_feature_i), "common points between X_3d_Chirality_OK and ", i, " image")
            continue

        pts_i = np.hstack((feature_x[id_feature_i, i].reshape(-1,1), feature_y[id_feature_i, i].reshape(-1,1)))
        X_3d_Correspondance = X_3d_all[id_feature_i, :].reshape(-1,3)
        
        
        # PnP - estimating camera pose given 3d world and 2d prokjected points
        R_init, C_init = PnPRANSAC(K, pts_i, X_3d_Correspondance, n_itr = 1000, error_thresh = 5)
        LinearPnP_error = reprojectionErrorPnP(X_3d_Correspondance, pts_i, K, R_init, C_init)
        
        R_i, C_i = NonLinearPnP(K, pts_i, X_3d_Correspondance, R_init, C_init)
        errorNonLinearPnP = reprojectionErrorPnP(X_3d_Correspondance, pts_i, K, R_i, C_i)
        print("Error after linear PnP: ", LinearPnP_error, " Error after non linear PnP: ", errorNonLinearPnP)

        error_row_for_chart[0] = LinearPnP_error
        error_row_for_chart[1] = errorNonLinearPnP

        C_reg_set_.append(C_i)
        R_reg_set_.append(R_i)
        


        # Applying Trianglulation
        for j in range(0, i+1):
        
            idx_X_pts = np.where(feature_flag_filtered[:, j] & feature_flag_filtered[:, i])
            if (len(idx_X_pts[0]) < 8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts, j].reshape((-1, 1)), feature_y[idx_X_pts, j].reshape((-1, 1))))
            x2 = np.hstack((feature_x[idx_X_pts, i].reshape((-1, 1)), feature_y[idx_X_pts, i].reshape((-1, 1))))

            X = LinearTriangulation(K, C_reg_set_[j], R_reg_set_[j], C_i, R_i, x1, x2)
            X = X/X[:,3].reshape(-1,1)
            
            Error_LT = meanReprojectionError(X, x1, x2, R_reg_set_[j], C_reg_set_[j], R_i, C_i, K)
            
            X_3d_all[idx_X_pts] = X[:,:3]
            X_found_flag[idx_X_pts] = 1
            np.save(savepath+'No_BA/optimized_NoBA_C_set_T'+ str(i) + str(j), C_reg_set_)
            np.save(savepath+'No_BA/optimized_NoBA_R_set_T'+ str(i)+ str(j), R_reg_set_)
            np.save(savepath+'No_BA/optimized_NoBA_X_all_T'+ str(i)+ str(j), X_3d_all)
            np.save(savepath+'No_BA/optimized_NoBA_X_found_T'+ str(i)+ str(j), X_found_flag)

            #Applying Non Linear Triangulation
            X = NonLinearTriangulation(K, x1, x2, X, R_reg_set_[j], C_reg_set_[j], R_i, C_i)
            X = X/X[:,3].reshape(-1,1)
            
            nLT_error = meanReprojectionError(X, x1, x2, R_reg_set_[j], C_reg_set_[j], R_i, C_i, K)
            print("Error after linear triangulation: ", Error_LT, " Error after non linear triangulation: ", nLT_error)
            
            error_row_for_chart[2 + j] = Error_LT
            error_row_for_chart[8 + j] = nLT_error

            X_3d_all[idx_X_pts] = X[:,:3]
            X_found_flag[idx_X_pts] = 1
            
            print("appended ", len(idx_X_pts[0]), " points between ", j ," and ", i)
            np.save(savepath+'No_BA/optimized_NoBA_C_set_NT'+ str(i) + str(j), C_reg_set_)
            np.save(savepath+'No_BA/optimized_NoBA_R_set_NT'+ str(i) + str(j), R_reg_set_)
            np.save(savepath+'No_BA/optimized_NoBA_X_all_NT'+ str(i) + str(j), X_3d_all)
            np.save(savepath+'No_BA/optimized_NoBA_X_found_NT'+ str(i) + str(j), X_found_flag)
        
        
        #Performaing Bundle Adjustment
        print( 'Performing Bundle Adjustment  for image : ', i  )
        R_reg_set_, C_reg_set_, X_3d_all = BundleAdjustment(X_3d_all,X_found_flag, feature_x, feature_y,
                                                    feature_flag_filtered, R_reg_set_, C_reg_set_, K, nCam = i)
        
    
        for k in range(0, i+1):
            idx_X_pts = np.where(X_found_flag[:,0] & feature_flag_filtered[:, k])
            x = np.hstack((feature_x[idx_X_pts, k].reshape((-1, 1)), feature_y[idx_X_pts, k].reshape((-1, 1))))
            X = X_3d_all[idx_X_pts]
            BA_error = reprojectionErrorPnP(X, x, K, R_reg_set_[k], C_reg_set_[k])
            print("Error after BA :", BA_error)
            error_row_for_chart[14+k] = BA_error
            np.save(savepath+'BA/optimized_C_set_'+ str(i) + str(j), C_reg_set_)
            np.save(savepath+'BA/optimized_R_set_'+ str(i) + str(j), R_reg_set_)
            np.save(savepath+'BA/optimized_X_all'+ str(i) + str(j), X_3d_all)
            np.save(savepath+'BA/optimized_X_found'+ str(i) + str(j), X_found_flag)
        print('##################### Registered Camera : ', i+1, '######################')
        error_chart.writerow(list(error_row_for_chart))
        print("RRRR Error after BA :", BA_error)
        
    X_found_flag[X_3d_all[:,2]<0] = 0    
    print('##########################################################################')

#See_save_plot()    

   
    
if __name__ == '__main__':
    main()
         

        
