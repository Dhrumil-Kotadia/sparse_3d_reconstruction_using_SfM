import numpy as np
from matplotlib import pyplot as plt

# C_set_ = np.load('../outputs/BA/optimized_C_set_.npy')
# R_set_ = np.load('../outputs/BA/optimized_R_set_.npy')
# X_all_ = np.load('../outputs/BA/optimized_X_all.npy')
# X_found_ = np.load('../outputs/BA/optimized_X_found.npy')

# C_set_before = np.load('../outputs/No_BA/optimized_NoBA_C_set_.npy')
# R_set_before = np.load('../outputs/No_BA/optimized_NoBA_R_set_.npy')
# X_all_before = np.load('../outputs/No_BA/optimized_NoBA_X_all.npy')
# X_found_before = np.load('../outputs/No_BA/optimized_NoBA_X_found.npy')


# C_set_ = np.load('../outputs/No_BA_new/optimized_NoBA_C_set_NT22.npy')
# R_set_ = np.load('../outputs/No_BA_new/optimized_NoBA_R_set_NT22.npy')
# X_all_ = np.load('../outputs/No_BA_new/optimized_NoBA_X_all_NT22.npy')
# X_found_ = np.load('../outputs/No_BA_new/optimized_NoBA_X_found_NT22.npy')

# C_set_ = np.load('../outputs/No_BA_new/optimized_NoBA_C_set_T23.npy')
# R_set_ = np.load('../outputs/No_BA_new/optimized_NoBA_R_set_T23.npy')
# X_all_ = np.load('../outputs/No_BA_new/optimized_NoBA_X_all_T23.npy')
# X_found_ = np.load('../outputs/No_BA_new/optimized_NoBA_X_found_T23.npy')

C_set_before = np.load('../outputs/No_BA_new/optimized_NoBA_C_set_NT23.npy')   
R_set_before = np.load('../outputs/No_BA_new/optimized_NoBA_R_set_NT23.npy')
X_all_before = np.load('../outputs/No_BA_new/optimized_NoBA_X_all_NT23.npy')
X_found_before = np.load('../outputs/No_BA_new/optimized_NoBA_X_found_NT23.npy')

C_set_ = np.load('../outputs/BA/optimized_C_set_23.npy')
R_set_ = np.load('../outputs/BA/optimized_R_set_23.npy')
X_all_ = np.load('../outputs/BA/optimized_X_all23.npy')
X_found_ = np.load('../outputs/BA/optimized_X_found23.npy')


savepath = '../outputs/'
fig = plt.figure(figsize = (15,10))

feature_idx = np.where(X_found_before[:, 0])
X_before = X_all_before[feature_idx]
x1 = X_before[:,0]
z1 = X_before[:,2]
x1[(x1 < -500) | (x1 > 500)] = 0 
z1[(z1 <= 0) | (z1 > 500)] = -z1[(z1 <= 0) | (z1 > 500)]

plt.xlim(-250,  250)
plt.ylim(0,  500)
plt.scatter(x1, z1, marker='x',linewidths=0.5, color = 'green')


feature_idx = np.where(X_found_[:, 0])
X = X_all_[feature_idx]

x = X[:,0]
z = X[:,2]
x[(x < -500) | (x > 500)] = 0 
z[(z <= 0) | (z > 500)] = -z[(z <= 0) | (z > 500)]

plt.xlim(-250,  250)
plt.ylim(-250,  250)
plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
plt.savefig(savepath+'2D.png')
plt.show()
