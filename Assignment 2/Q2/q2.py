# %% [markdown]
# # Question 2

# %%
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import open3d as o3d

# %% [markdown]
# ## 2.1

# %% [markdown]
# ### Lists defined to store the numericals values 

# %%
lidar_normals = np.empty((35, 3))
lidar_offsets = []
lidar_points = []
camera_normals = np.empty((35, 3)) # (35 x 3)
camera_intrinsic_matrix = np.zeros((3, 3))
distortion_coefficients = np.zeros((1, 5))
rotation_matrices = np.empty((35, 3, 3)) # (35 x 3 x 3)
translation_vectors = np.empty((35, 3)) # (35 x 3)
rotation_vectors = np.empty((35, 3)) # (35 x 3)
corner_points = []
centroids = []
normals_lidar_frame = []
offsets_lidar_frame = []

# %% [markdown]
# ### Getting Lidar Normals and Lidar Offsets

# %%
source_folder = "datasets\lidar_scans"
extension = "*.pcd"

files = sorted(glob.glob(os.path.join(source_folder, extension)))
ctr = 0

for fname in files:
    # Loading the point cloud data from the PCD file using Open3D
    pcd = o3d.io.read_point_cloud(fname)

    # Extracting the planar LIDAR points using RANSAC
    model_planar, inl = pcd.segment_plane(num_iterations=100, ransac_n=3, distance_threshold=0.01)

    # Computing the centroid of the planar points
    plane_points = np.asarray(pcd.points)[inl]

    lidar_points.append(plane_points)
    
    #computing the centroids of all the lidar points
    centroid = np.mean(plane_points, axis=0)
    
    #storing it in an array
    centroids.append(centroid)
    plane_points_centered = plane_points - centroid
    
    # Covariance matrix
    transposed_ppc = plane_points_centered.T
    covar_matrix = np.cov(transposed_ppc)
    
    # Computing SVD
    unit_array, sing_values, matrix = np.linalg.svd(plane_points_centered)

    # Offset of the chessboard plane is be computed by dot product of the normal and the centroid
    chessboard_offset = -np.dot(chessboard_normal, centroid)
    
    # The last column of the matrix is the chessboard_normal
    chessboard_normal = matrix[2]
    chessboard_normal = np.array(chessboard_normal)
    
    hom_normal_lidar = np.hstack((chessboard_normal, chessboard_offset))
    vec_norm = np.linalg.norm(hom_normal_lidar[:3])
    
    hom_normal_lidar[:3] = hom_normal_lidar[:3] * (108 / vec_norm)
    norm = hom_normal_lidar[:3]
    off = hom_normal_lidar[3]
    
    normals_lidar_frame.append(norm)
    offsets_lidar_frame.append(off)
    
    
    lidar_normals[ctr] = chessboard_normal
    lidar_offsets.append(chessboard_offset) 
    print("Chessboard Plane Normals (nL) for lidar no", ctr+1, "is : \n", chessboard_normal)
    print("Corresponding offsets for lidar no", ctr+1, "is : \n", chessboard_offset)
    print()
    ctr += 1
    
offsets_lidar_frame = np.array(offsets_lidar_frame)
centroids = np.array(centroids)



# %% [markdown]
# ### Getting Camera Normals

# %%
path = "datasets/camera_parameters"

foldernames = sorted(os.listdir(path))
ctr = 0
for foldername in foldernames:
    if os.path.isdir(os.path.join(path, foldername)):
        normals_file = os.path.join(path, foldername, "camera_normals.txt")
        if os.path.isfile(normals_file):
            with open(normals_file, "r") as f:
                x = []
                for line in f:
                    for n in line.strip().split():
                        x.append(float(n))
                x = np.array(x)
                camera_normals[ctr] = x
                ctr += 1
                
print("Camera normals are: \n", camera_normals)


# %% [markdown]
# ### Getting Camera Intrinsic Matrix

# %%
path = "datasets/camera_intrinsic.txt"
with open(path, 'r') as f:
    content = f.readlines()

for i, line in enumerate(content):
    values = line.split()
    camera_intrinsic_matrix[i] = [float(val) for val in values]
    
print("The camera intrinsic matrix is: \n", camera_intrinsic_matrix)

# %% [markdown]
# ### Getting Distortion Coefficients

# %%
path = "datasets/distortion.txt"
with open(path, 'r') as f:
    content = f.readlines()

for i, line in enumerate(content):
    values = line.split()
    distortion_coefficients[i] = [float(val) for val in values]
distortion_coefficients = distortion_coefficients.reshape(5,)

# %% [markdown]
# ### Getting Rotation Matrices, Rotation Vectors, Translation Vectors and Camera Normals

# %%
path = "datasets/camera_parameters"
foldernames = sorted(os.listdir(path))
ctr = 0
for foldername in foldernames:
    if os.path.isdir(os.path.join(path, foldername)):
        rotation_file = os.path.join(path, foldername, "rotation_matrix.txt")
        if os.path.isfile(rotation_file):
            with open(rotation_file, 'r') as f:
                content = f.readlines()

            rot_matrix = np.zeros((3, 3))
            for i, line in enumerate(content):
                values = line.split()
                rot_matrix[i] = [float(val) for val in values]
            rotation_matrices[ctr] = rot_matrix
                
        translation_file = os.path.join(path, foldername, "translation_vectors.txt")
        if os.path.isfile(translation_file):
            with open(translation_file, "r") as f:
                x = []
                for line in f:
                    for n in line.strip().split():
                        x.append(float(n))
                translation_vectors[ctr] = np.array(x)
                
        rotation_file = os.path.join(path, foldername, "rotation_vectors.txt")
        if os.path.isfile(rotation_file):
            with open(rotation_file, "r") as f:
                x = []
                for line in f:
                    for n in line.strip().split():
                        x.append(float(n))
                rotation_vectors[ctr] = np.array(x)
                
        cam_normal_file = os.path.join(path, foldername, "camera_normals.txt")
        if os.path.isfile(cam_normal_file):
            with open(cam_normal_file, "r") as f:
                x = []
                for line in f:
                    for n in line.strip().split():
                        x.append(float(n))
                camera_normals[ctr] = np.array(x)
    ctr += 1


# %% [markdown]
# ### Getting corner points

# %%
source_folder = "datasets\camera_images"
extension = "*.jpeg" 
CHECKERBOARD = (8, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Extracting path of individual image stored in a given directory
files = sorted(glob.glob(os.path.join(source_folder, extension)))
ctr = 1

for fname in files:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corner = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        print("Corners recognised for image no: ", ctr)
        corners = cv2.cornerSubPix(gray, corner, (11,11),(-1,-1), criteria)
        corner_points.append(corners)
    else:
        print("Corners not recognised for image no: ", ctr)
        print("Corners not recognised for image: ", fname)
    ctr+=1


# %% [markdown]
# # 2.2

# %%
# Least-squares solution for the rotation matrix
lid_normal = np.matmul(lidar_normals.T, lidar_normals)
lidar_camera_normal = np.matmul(lidar_normals.T, camera_normals)
inverse_lidar_normal = np.linalg.inv(lid_normal)
lidar_camera_rotation = np.matmul(inverse_lidar_normal, lidar_camera_normal)

# computing centroid of lidar points on chessboard plane
centroids = np.mean(centroids, axis=0)

# computing translation vector
lidar_camera_translation_vec = np.matmul(-lidar_camera_rotation, centroids)

# compute the transformation matrix
twoDvector = np.atleast_2d(lidar_camera_translation_vec).T
lidar_camera_transformation_matrix = np.hstack((lidar_camera_rotation, twoDvector))

print("The camera to lidar transformation matrix is: \n\n", lidar_camera_transformation_matrix)

# %% [markdown]
# ## 2.3

# %% [markdown]
# ### Function to Estimate the lidar camera transform

# %%
def lc_transform(cam_normals, lid_normals):
    
    # Generating vector perpendicular to both camera normals and lidar normals
    nor_vec = np.dot(cam_normals.T, lid_normals)
    
    # Computing the rotation matrix
    rotation_mat, not_req = np.linalg.qr(nor_vec)
    
    #Computing determinant of the rotation matrix
    determinant = np.linalg.det(rotation_mat)

    # Compute the translation vector t
    translation_vec = np.mean(cam_normals, axis=0)
    
    if determinant < 0:
        rotation_mat[:, -1] = (-1) * rotation_mat[:, -1] 
        
    translation_vec -= np.matmul(rotation_mat, np.mean(lid_normals, axis=0))
    
    # Generating the final transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:-1, -1] = translation_vec
    transformation_matrix[:-1, :-1] = rotation_mat
    
    return transformation_matrix

# %%
print("The transformation matrix is: \n")
transformation_matrix = lc_transform(lidar_normals, lidar_normals)
print(transformation_matrix)
print("\nThe determinant of the transformation matrix is:", np.linalg.det(transformation_matrix))

# %% [markdown]
# ## 2.4

# %% [markdown]
# ###  Projecting LIDAR points to the image plane using the intrinsic camera parameters

# %%
source_folder_cam_images = "datasets\camera_images"
extension_cam_images = "*.jpeg" 

source_folder_lidar_scans = "datasets\lidar_scans"
extension_lidar_scans = "*.pcd" 

files_cam_images = sorted(glob.glob(os.path.join(source_folder_cam_images, extension_cam_images)))
files_lidar_scans = sorted(glob.glob(os.path.join(source_folder_lidar_scans, extension_lidar_scans)))

for i in range (len(files_cam_images)):
    # Load image
    cam_image = cv2.imread(files_cam_images[i])

    # Load LIDAR point cloud and transform to camera frame of reference
    pcd_cloud = o3d.io.read_point_cloud(files_lidar_scans[i])
    
    # Transforming the lidar points using the transformation matrix
    pcd_cloud.transform(transformation_matrix)

    # Converting pcd cloud points to numpy array
    pcd_pts = np.array(pcd_cloud.points)
    
    # Getting the projection points
    project_points, _ = cv2.projectPoints(pcd_pts, rotation_vectors[i], translation_vectors[i], camera_intrinsic_matrix, distortion_coefficients)

    
    # Drawing the projected points acquired onto the image
    marginal_error = 4
    for pts in range(len(project_points)):
        iter = 1
        for _, corner_pt in enumerate(corners[i]):
            coordinates_tup = tuple(map(int, project_points[pts][0]))
            x = corner_pt[0]
            y = corner_pt[1]
            
            a = x - marginal_error
            b = project_points[pts][0][0]
            c = x + marginal_error
            d = y - marginal_error
            e = project_points[pts][0][1]
            f = y + marginal_error
            
            if a <= b and b <= c and d <= e and e <= f:
                cv2.circle(cam_image, coordinates_tup, 6, (0, 0, 255), 40)
            else:
                print("A point is detected, outside the chessboard boundary.")
                
            iter += 1

      
    window_name = "Projected Image no " + str(i+1)
    plt.title(window_name)
    plt.imshow(cam_image)
    plt.show()

# %% [markdown]
# ## 2.5

# %% [markdown]
# ### Cosine distances

# %%
cosine_dis = []

for i in range (len(camera_normals)):
    cam_to_lid_matrix = transformation_matrix
    cam_to_lid_matrix = cam_to_lid_matrix[:3, :3]
    normal_lidar = np.matmul(cam_to_lid_matrix, lidar_normals[i])
    
    # Computing the cosine distances
    A = np.matmul(camera_normals[i], normal_lidar)
    B = np.linalg.norm(camera_normals[i])
    C = np.linalg.norm(normal_lidar)
    
    cosD = np.abs(A / (B*C))
    cosine_dis.append(cosD)
    
print("The cosine distances between the camera normal and the transformed LIDAR normals are: \n")
for i in cosine_dis:
    print(i)
    

# %% [markdown]
# ### Histogram

# %%
H = np.histogram(cosine_dis, bins = 35)

# %% [markdown]
# ### Plotting histogram errors

# %%
plt.hist(H, bins = 35)
plt.xlabel('Cosine Distances between camera normal and transformed LIDAR normals')
plt.legend(['Lidar Normals', 'Camera Normals'], loc = 'upper right')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# %% [markdown]
# ### Average error and Standard Deviation

# %%
error = np.mean(cosine_dis)
std_deviation = np.std(cosine_dis)

print("Average error:", error)
print("Standard Deviation", std_deviation)


