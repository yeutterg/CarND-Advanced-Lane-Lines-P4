import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters
k_size = 3 # Kernel size

camera_cal_dir = './camera_cal'
test_img_dir = './test_images'
out_img_dir = './output_images'

# Calibration
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image space

"""
1. Image Transformations
"""

def grayscale(img):
    """
    Converts a BGR image to grayscale

    img: The image in BGR color format
    return The grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hls(img):
    """
    Converts a BGR image to HLS

    img: The image in BGR color format
    return The HLS image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def undistort(img, gray):
    """
    Undistorts a camera image

    img: The image in color
    gray: The image in grayscale
    return The undistorted image
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)

def perspective_transform(img):
    """
    Transforms the perspective of the image

    img: The image to transform
    return The warped image
    """
    # Define the four source points
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])

    # Define the four destination points
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
    
    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Get the image size
    img_size = (img.shape[1], img.shape[0])

    # Warp the image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the warped image and transform matrix
    return warped, M

"""
2. Threshold Calculations
"""

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Computes the direction of the gradient in both the x and y directions
    and applies a threshold as a layer mask

    gray: The image to process, in grayscale
    sobel_kernel: The Sobel kernel size
    thres: The threshold to apply
    return The layer mask
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    # Return the mask
    return binary_output

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    """
    Computes the magnitude of the gradient and applies a threshold 
    as a layer mask

    gray: The image to process, in grayscale
    sobel_kernel: The Sobel kernel size
    thres: The threshold to apply
    return The layer mask
    """
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_factor = np.max(mag_sobel)/255 
    mag_scaled = (mag_sobel/scale_factor).astype(np.uint8) 

    # Create a mask of 1s where the scaled gradient magnitude 
    # is > thresh_min and <= thresh_max
    binary_output = np.zeros_like(mag_scaled)
    binary_output[(mag_scaled >= thresh[0]) & (mag_scaled <= thresh[1])] = 1

    # Return the mask
    return binary_output

def abs_sobel_thresh(gray, orient='x', thresh=(0, 255), sobel_kernel=3):
    """
    Computes the absolute value of Sobel in the x or y direction
    and applies a threshold as a layer mask

    gray: The image to process, in grayscale
    orient: The orientation, either 'x' or 'y'
    sobel_kernel: The Sobel kernel size
    thres: The threshold to apply
    return The layer mask
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y')

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Create a mask of 1s where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    # 6) Return the mask
    return binary_output

def saturation_thresh(img, thresh=(0, 255)):
    """
    Computes the threshold of the saturation of the image

    img: The image to process, in BGR
    thres: The threshold to apply
    return The layer mask
    """
    # Convert the image to HLS format
    hlsimg = hls(img)

    # Apply the threshold to the S channel
    S = hlsimg[:,:,2]

    # Create a mask with the threshold applies
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # Return the mask
    return binary

"""
3. Data Processing
"""

def calibrate_chessboard(xdim=9, ydim=6, drawCorners=0, saveFile=0):
    """
    Calibrates chessboard images by correcting lens distortions

    xdim: The number of corners in the x axis
    ydim: The number of corners in the y axis
    drawCorners: Boolean. If true, draws detected corners on image
    saveFile: Boolean. If true, saves a comparison of the original and 
              undistorted image
    return An array of calibrated images and the warping source array
    """
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
            undist = undistort(img, gray)

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

    # Return the output images and src array           
    return out

# For calibrateChessboard function demonstration, saves output with lines
# calibrate_chessboard(drawCorners=1, saveFile=1) 

def img_process_pipeline(fname, ksize=3, saveFile=0):
    """
    The pipeline for image processing

    fname: The filename of the image to process
    ksize: The kernel size
    src: The source array of detected corners from the calibration step
    saveFile: Boolean: If true, saves the image at various steps along
              the pipeline
    return The processed image
    """
    # Get the image from the filename
    img = mpimg.imread(fname)

    # Convert the image to grayscale
    gray = grayscale(img)

    # Correct image distortion
    undist = undistort(img, gray)
    if saveFile:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_img_dir + '/undist_' + fname.split('/')[-1])

    # Get the grayscale of the undistorted image
    gray = grayscale(undist)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    sat_binary = saturation_thresh(undist, thresh=(200, 255))

    # Combine the thresholding results
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))| (sat_binary == 1)] = 1
    if saveFile:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist)
        ax1.set_title('Undistorted Image', fontsize=50)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_img_dir + '/combined_' + fname.split('/')[-1])

    # Perspective transform (bird's eye view)
    warped, M = perspective_transform(img)
    if saveFile:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist)
        ax1.set_title('Undistorted Image', fontsize=50)
        ax2.imshow(warped)
        ax2.set_title('Undist. & Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_img_dir + '/persp_' + fname.split('/')[-1])


def main():
    """
    The main project pipeline
    """
    # Calibrate the camera lens using chessboard images
    calibrate_chessboard()

    # Correct a single image for distortion
    fname = test_img_dir + "/test1.jpg"
    img_process_pipeline(fname, ksize=k_size, saveFile=1)

main()
