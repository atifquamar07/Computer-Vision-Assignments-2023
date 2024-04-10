# %% [markdown]
# ## Computer Vision - Assignment 2

# %% [markdown]
# ## 1.1

# %%
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# %%
# Defining the dimensions of checkerboard
CHECKERBOARD = (7,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# %%
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# %%
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# %%
source_folder = "images"
extension = "*.jpg" 

# %%
# Extracting path of individual image stored in a given directory
files = glob.glob(os.path.join(source_folder, extension))
ctr = 1

for fname in files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        print("Corners recognised for image no: ", ctr)
        objpoints.append(objp)
        corners = cv2.cornerSubPix(gray, corner, (11,11),(-1,-1), criteria)
        imgpoints.append(corners)

    ctr+=1



# %%
# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# %%
# Print intrinsic parameters and error estimates
print("Camera matrix: \n", mtx)
print("Focal length along X-axis: ", mtx[0, 0], " ± ", ret)
print("Focal length along Y-axis: ", mtx[1, 1], " ± ", ret)
print("Skew: ", mtx[0, 1], " ± ", ret)
print("Principal point: ","(", mtx[0, 2], ", ", mtx[1, 2], ")", " ± ","(", ret, ", ", ret, ")")

# %% [markdown]
# ## 1.2

# %%
image_no = 1
files = glob.glob(os.path.join(source_folder, extension))
for fname in files:
    img = cv2.imread(fname)
    # Detect the corners of the chessboard in the image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

    # Estimate the extrinsic camera parameters
    ret, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Print the rotation matrix and translation vector
    print('Rotation matrix for image no', image_no, ': ')
    print(R)
    print('Translation vector for image no', image_no, ': ')
    print(tvec)
    print()
    image_no+=1

# %% [markdown]
# ## 1.3
# 

# %% [markdown]
# #### Radial Distortion Coefficients

# %%
# Print out radial distortion coefficients
print("Radial distortion coefficients: ", dist)

# %%
image_no = 1
files = glob.glob(os.path.join(source_folder, extension))
for fname in files:
    if (image_no >= 6):
        break
    # Load raw image
    img_raw = cv2.imread(fname)

    # # Undistort image
    # img_undistorted = cv2.undistort(img_raw, mtx, dist, cv2.COLOR_BGR2RGB)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_raw.shape[1::-1], 1, img_raw.shape[1::-1])

    # Undistort the image using cv2.undistort()
    undistorted_img = cv2.undistort(img_raw, mtx, dist, cv2.COLOR_BGR2RGB, new_camera_matrix)
    
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_raw)
    ax[0].set_title("Original Image")
    ax[1].imshow(undistorted_img)
    ax[1].set_title("Undistorted Image")
    plt.show()
    image_no+=1
    

# %% [markdown]
# ## 1.4

# %% [markdown]
# ### Reprojection Errors

# %%
errors = []
for i in range (len(files)):
    project_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], project_points, cv2.NORM_L2) / len(project_points)
    errors.append(error)

# %%
# Plot the re-projection error for each image
plt.bar(range(1,len(errors)+1), errors)
plt.title("Re-projection Error for Each Image")
plt.xlabel("Image Number")
plt.ylabel("Error")
plt.show()

# Report the mean and standard deviation of the re-projection error
mean_error = np.mean(errors)
std_error = np.std(errors)
print("Mean re-projection error: {:.2f} pixels".format(mean_error))
print("Standard deviation of re-projection error: {:.2f} pixels".format(std_error))

# %% [markdown]
# ## 1.5

# %%
# Plot corners detected in the image and the re-projected corners for all images
ctr = 0
for i, file in enumerate(files):
    
    # Load image
    img = cv2.imread(file)

    # Find corners in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    # print(len(corners))
    
    # Draw corners detected in the image
    image_corners = img.copy()
    image_corners = cv2.drawChessboardCorners(image_corners, CHECKERBOARD, corners, ret)
    
    color = (0, 0, 255)
    for cor in corners:
        radius = 6
        center=tuple(map(int, cor[0]))
        cv2.circle(image_corners, center, radius, color, 40)

    # Re-project corners onto the image
    project_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Draw re-projected corners onto the image
    projected_corners = img.copy()
    projected_corners = cv2.drawChessboardCorners(projected_corners, CHECKERBOARD, project_points, True)
    
    for cor in project_points:
        radius = 6
        center=tuple(map(int, cor[0]))
        cv2.circle(projected_corners, center, radius, color, 40)

    # Plot the two images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(image_corners, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Detected Corners")
    ax[1].imshow(cv2.cvtColor(projected_corners, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Re-projected Corners")
    plt.show()
    ctr+=1

# %% [markdown]
# ## 1.6

# %% [markdown]
# ### Computing the checkerboard plane normals

# %%
# Compute the checkerboard plane normals in the camera coordinate frame of reference
plane_normals = []
for i in range(len(objpoints)):
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvecs[i])

    # Compute the checkerboard plane normal in the camera coordinate frame
    normal = np.dot(R, np.array([0, 0, 1]))
    plane_normals.append(normal)

print("The plane normals for each image are: \n")
print()
for i in range (len(plane_normals)):
    print("For image ", i+1, ": ")
    print(plane_normals[i])
    print()


