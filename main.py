import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./images/z-dimension.csv")

# Load all images and Z coordinates
images = []
zs = []

for i in range(len(df)):
    images.append(cv.imread("./images/"+df['Image Name'][i] +".png",cv.IMREAD_GRAYSCALE))
    zs.append(df["Z-Cordinates (in mm)"][i])

# Find corners in Checker Board Images
crnr_coords = []
for img in images:
    corners = cv.goodFeaturesToTrack(img, 88, 0.01, 10)
    corners = np.intp(corners)

    crnr_coords.append(corners)
    
# Real world points X,Y
objpt = np.mgrid[0:176:25,0:251:25].reshape(88,2)
objpt = np.column_stack((objpt, zs[0]*np.ones(88)))

# print(len(crnr_coords))
# Camera(Image) points 
imgpt = crnr_coords[0].reshape(88,2)
imgpt = imgpt.T
sorted_indices = np.argsort(imgpt[0])  # Get sorted indices of the first row
imgpt = imgpt[:, sorted_indices].T

x,y = imgpt.T
y = y.reshape(11,8)
y = np.sort(y,axis = 1)
y = y.flatten()
imgpt = np.vstack((x.T,y.T)).T

# print(imgpt)
print(imgpt.shape)


# function to find A
def find_A(objpts,imgpts):
    assert objpts.shape[0] == imgpts.shape[0]
    A = np.empty((2*objpts.shape[0],12))
    # print(A.shape)
    for i,(wrld_pt,cam_pt) in enumerate(zip(objpts,imgpts)):
        Xi,Yi,Zi = wrld_pt
        xi,yi = cam_pt
        A[2*i] = np.array([Xi, Yi, Zi, 1, 0,  0,  0,  0, -xi*Xi, -xi*Yi, -xi*Zi, -xi])
        A[2*i + 1] = np.array([0,  0,  0,  0, Xi, Yi, Zi, 1, -yi*Xi, -yi*Yi, -yi*Zi, -yi])
    return A


A = (find_A(objpt,imgpt))

_,_,VT = np.linalg.svd(A)

M = VT[-1, :]
M = M.reshape(3,4)
print(M)