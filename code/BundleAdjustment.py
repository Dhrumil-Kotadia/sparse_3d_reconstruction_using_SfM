import numpy as np
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
from Utils.MiscUtils import *
from BuildVisibilityMatrix import *

def CameraPtIndices_get(visiblity_matrix):
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)


def _2D_pts_get(X_idx, visiblity_matrix, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_idx]
    visible_feature_y = feature_y[X_idx]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)             




def Sparsity_BA(X_found, feature_flag_filtered, nCam):
    
    cam_no = nCam + 1
    X_idx, visiblity_matrix = ObsIdxandVizMat(X_found.reshape(-1), feature_flag_filtered, nCam)
    n_obs = np.sum(visiblity_matrix)
    pts_n = len(X_idx[0])

    m = n_obs * 2
    n = cam_no * 6 + pts_n * 3
    A = lil_matrix((m, n), dtype=int)
    print(m, n)


    i = np.arange(n_obs)
    camera_indices, point_indices = CameraPtIndices_get(visiblity_matrix)

    for d in range(6):
        A[2 * i, camera_indices * 6 + d] = 1
        A[2 * i + 1, camera_indices * 6 + d] = 1

    for d in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + d] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + d] = 1

    return A


def project(_3d_pts, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = _3d_pts[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
   
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def function(x0, nCam, pts_n, camera_indices, point_indices, _2d_pts, K):
   
    cam_no = nCam + 1
    camera_params = x0[:cam_no * 6].reshape((cam_no, 6))
    _3d_pts = x0[cam_no * 6:].reshape((pts_n, 3))
    points_proj = project(_3d_pts[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - _2d_pts).ravel()
    
    return error_vec

def BundleAdjustment(X_all,X_found, feature_x, feature_y, feature_flag_filtered, R_set_, C_set_, K, nCam):
    
    X_idx, visiblity_matrix = ObsIdxandVizMat(X_found, feature_flag_filtered, nCam)
    _3d_pts = X_all[X_idx]
    _2d_pts = _2D_pts_get(X_idx, visiblity_matrix, feature_x, feature_y)

    list_RC = []
    print('R_set_:', R_set_)
    for i in range(nCam+1):
        C, R = C_set_[i], R_set_[i]
        Q = getEuler(R)
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        list_RC.append(RC)
    list_RC = np.array(list_RC).reshape(-1, 6)

    x0 = np.hstack((list_RC.ravel(), _3d_pts.ravel()))
    pts_n = _3d_pts.shape[0]

    camera_indices, point_indices = CameraPtIndices_get(visiblity_matrix)
    
    A = Sparsity_BA(X_found, feature_flag_filtered, nCam)
    t0 = time.time()
    res = least_squares(function, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, pts_n, camera_indices, point_indices, _2d_pts, K))
    t1 = time.time()
    print('Run BA :', t1-t0, 's \nA matrix shape: ' ,  A.shape, '\n')
    
    x1 = res.x
    cam_no = nCam + 1
    print('Optimized camera parameters: \n', x1[:cam_no * 6].reshape((cam_no, 6)))
    camera_params_opt = x1[:cam_no * 6].reshape((cam_no, 6))
    _3d_pts_opt = x1[cam_no * 6:].reshape((pts_n, 3))

    X_all_optzed = np.zeros_like(X_all)
    X_all_optzed[X_idx] = _3d_pts_opt

    C_set_optzed, R_set_optzed = [], []
    for i in range(len(camera_params_opt)):
        R = getRotation(camera_params_opt[i, :3], 'e')
        C = camera_params_opt[i, 3:].reshape(3,1)
        C_set_optzed.append(C)
        R_set_optzed.append(R)
    
    return R_set_optzed, C_set_optzed, X_all_optzed


