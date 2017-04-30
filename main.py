import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

camera_cal_dir = './camera_cal'
test_img_dir = './test_images'
out_img_dir = './output_images'

def grayscale(img):
    """
    Converts an image to grayscale

    img: The image in BGR color format
    return The grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def undistort(img, gray, objpoints, imgpoints):
    """
    Undistorts a camera image

    img: The image in color
    gray: The image in grayscale
    objpoints: Array of 3D points in real world space
    imgpoints: Array of 2D Points in image space
    return The undistorted image
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)

def calibrateChessboard(xdim=9, ydim=6, drawCorners=0, saveFile=0):
    """
    Calibrates chessboard images by correcting lens distortions

    xdim: The number of corners in the x axis
    ydim: The number of corners in the y axis
    drawCorners: Boolean. If true, draws detected corners on image
    saveFile: Boolean. If true, saves a comparison of the original and 
              undistorted image
    return An array of calibrated images
    """
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image space
    out = [] # Output images

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... (7,5,0)
    objp = np.zeros((ydim*xdim, 3), np.float32)
    objp[:,:2] = np.mgrid[0:xdim,0:ydim].T.reshape(-1,2) # x, y coordinates

    # Import the calibration images
    images = glob.glob(camera_cal_dir + '/calibration*.jpg') 

    # Process each calibration image
    for fname in images:
        # Read in the images
        img = mpimg.imread(fname)

        # Convert to grayscale
        gray = grayscale(img) 

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (xdim, ydim), None)

        # If corners found
        if ret:
            # Add object and image points
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw the corners
            if drawCorners:
                img = cv2.drawChessboardCorners(img, (xdim, ydim), corners, ret)
            
            # Undistort
            undist = undistort(img, gray, objpoints, imgpoints)

            # Append to the output
            out.append(undist)

            # Save the undistorted image
            if saveFile:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=50)
                ax2.imshow(undist)
                ax2.set_title('Undistorted Image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.savefig(out_img_dir + '/out_' + fname.split('/')[-1])

    # Return the output images            
    return out

# For function demonstration, saves output with lines
# calibrateChessboard(drawCorners=1, saveFile=1) 