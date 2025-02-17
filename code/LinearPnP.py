import numpy as np
from Utils.MiscUtils import ProjectionMatrix, homography

def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = ProjectionMatrix(R,C,K)
    # print("P", P)
    Error = []
    # print("x3D", x3D) 
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = homography(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error

def PnP(X_set, x_set, K):
    # X_set is the 3D points
    N = X_set.shape[0]
    # x_set is the 2D points
    _4_X = homography(X_set)
    _3_x = homography(x_set)
    
    # normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(_3_x.T).T
    #  x_n = _3_x
    for i in range(N):
        X = _4_X[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_n[i]
        # print("u, v", u, v)
        crs_u = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        tilde_X = np.vstack((np.hstack((   X, zeros, zeros)), 
                            np.hstack((zeros,     X, zeros)), 
                            np.hstack((zeros, zeros,     X))))
        a = crs_u.dot(tilde_X)
        
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    _, _, VT = np.linalg.svd(A)
    P = VT[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_rT = np.linalg.svd(R) # to enforce Orthonormality
    R = U_r.dot(V_rT)
    # print("R", R)
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C


