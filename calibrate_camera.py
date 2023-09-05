import numpy as np
import cv2
import glob
import json
import argparse
import os
import time

def is_dir(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
    
def get_points_from_dir(dirpath):
    objpoints = []
    imgpoints = []
    
    imagetypes = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG')
    img_paths = []
    for img_t in imagetypes:
        img_paths.extend(glob.glob(os.path.join(dirpath, img_t)))
    
    for img_path in img_paths:
        gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        ret, corners = cv2. findChessboardCorners(gray, (CORNER_W, CORNER_H), None)
        
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
    imgwh = gray.shape[::-1]
        
        
    return objpoints, imgpoints, imgwh

def get_points_online():
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    while(True):
        ret, frame = cap.read()
        # cv2.imshow('frame', frame)
        
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_corner = np.copy(gray)
        
        ret, corners = cv2.findChessboardCorners(gray, (CORNER_W, CORNER_H), None)
        
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            
            cv2.drawChessboardCorners(gray_corner, (CORNER_W, CORNER_H), corners2, ret)
        
        draw_frame = np.concatenate((gray, gray_corner), axis=1)
        cv2.imshow('calibrate', draw_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cv2.destroyAllWindows()
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    
    imgwh = gray.shape[::-1]
    
    return objpoints, imgpoints, imgwh


parser = argparse.ArgumentParser(description='Calibrate camera.')
parser.add_argument('fpath', nargs='?', metavar='file or directory path', type=is_dir)
args = parser.parse_args()
print(args)

fpath = args.fpath
IS_ONLINE = fpath == None
OUT_JSON = 'calibration_results.json'


# setup parameter

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

CORNER_H = 6
CORNER_W = 9

objp = np.zeros((CORNER_H*CORNER_W, 3), np.float32)
objp[:,:2] = np.mgrid[0:CORNER_W, 0:CORNER_H].T.reshape(-1,2)


# calibrate

t_start = time.time()

if IS_ONLINE:
    objpoints, imgpoints, imgwh = get_points_online()
else:
    objpoints, imgpoints, imgwh = get_points_from_dir(fpath)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgwh, None, None)

t_finish = time.time()

"""
file should start with 
{
    "calibration_results" : [

    ]
}
"""

with open(OUT_JSON, 'r+') as f:
    file_data = json.load(f)
    file_data['calibration_results'].append({
        'fpath' : str(os.path.abspath(fpath)) if fpath else 'online',
        'nframe' : len(imgpoints),
        'ret' : ret,
        'mtx' : mtx.tolist(),
        'dist' : dist.tolist(),
        'rvecs' : [vec.tolist() for vec in rvecs],
        'tvecs' : [vec.tolist() for vec in tvecs]
    })
    f.seek(0)
    json.dump(file_data, f, indent=4)

print('finished.')
print(f'process {len(imgpoints)} frames.')
print(f'take {t_finish - t_start} sec.')
print(f'saving result to {OUT_JSON}.')



print((
    ret,
    mtx,
    dist,
    rvecs,
    tvecs
))



