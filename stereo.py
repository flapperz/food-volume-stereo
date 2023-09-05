import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
    '''

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
        
L_FPATH = '/Users/flap/MasterThesis/source/food-volume-stereo/calibrate_images/ip12_image_small/IMG_7161.jpeg'
R_FPATH = '/Users/flap/MasterThesis/source/food-volume-stereo/calibrate_images/ip12_image_small/IMG_7163.jpeg'

image_left = cv2.imread(L_FPATH)
image_right = cv2.imread(R_FPATH)

image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

output_file = "out.ply"

if image_left.shape[0] != image_right.shape[0] or \
        image_left.shape[1] != image_right.shape[1]:
    raise TypeError("Input images must be of the same size")

# downscale images for faster processing
image_left = cv2.pyrDown(image_left)  
image_right = cv2.pyrDown(image_right)

image_left = cv2.pyrDown(image_left)  
image_right = cv2.pyrDown(image_right)

image_left = cv2.pyrDown(image_left)  
image_right = cv2.pyrDown(image_right)

# disparity range is tuned for 'aloe' image pair
win_size = 1
min_disp = 16
max_disp = min_disp * 4
num_disp = max_disp - min_disp   # Needs to be divisible by 16
stereo = cv2.StereoSGBM(minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = win_size,
    uniquenessRatio = 10,
    speckleWindowSize = 20,
    speckleRange = 16,
    disp12MaxDiff = 1,
    P1 = 8*3*win_size**2,
    P2 = 32*3*win_size**2,
    fullDP = True
)

# disparity_map = stereo.compute(image_left, image_right)
disparity_map = stereo.compute(image_left, image_right).astype(np.float32) / 16.0



print("\nComputing the disparity map ...")


# print "\nGenerating the 3D map ..."
h, w = image_left.shape[:2]
focal_length = 0.8*w                          

# Perspective transformation matrix
Q = np.float32([[1, 0, 0, -w/2.0],
                [0,-1, 0,  h/2.0], 
                [0, 0, 0, -focal_length], 
                [0, 0, 1, 0]])

points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

print("\nCreating the output file ...\n")
create_output(output_points, output_colors, output_file)