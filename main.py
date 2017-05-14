import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

# Parameters
k_size = 15  # Kernel size

camera_cal_dir = './camera_cal'
test_img_dir = './test_images'
out_img_dir = './output_images'

# Calibration
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image space

# Store fit values
left_fit_hist = []
right_fit_hist = []
left_fit_prev = []
right_fit_prev = []

"""
1. Image Transformations and Analysis
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)


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
    return (warped): The warped image, (M): The transform matrix,
            (Minv): The inverse transform matrix
    """
    # Define the four source points
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])

    # Define the four destination points
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])

    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Get the inverse transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Get the image size
    img_size = (img.shape[1], img.shape[0])

    # Warp the image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the warped image, transform matrix, and inverted transform matrix
    return warped, M, Minv


def histogram(img):
    """
    Gets the histogram of the bottom half of the image

    img: The image
    return The histogram
    """
    # Take a histogram of the bottom half of the image
    return np.sum(img[img.shape[0] // 2:, :], axis=0)


def sliding_window(img, histogram, nwindows=9, margin=100, minpix=50, saveFile=0, filename=''):
    """
    Implements a sliding window search to obtain a polynomial fit
    for each lane line

    img: The image
    histogram: The histogram of light points in the image
    nwindows: The number of sliding windows
    margin: The width of windows +/- this margin
    minpix: The minimum number of pixels to recenter a window
    saveFile: Whether to save an output figure
    filename: The name of the file if saving an output figure
    return: (left_fit) The left polynomial, (right_fit) The right polynomial
    """
    # Create an output image
    out_img = np.dstack((img, img, img)) * 255

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set the window height
    window_height = np.int(img.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions, to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Lists to store indices of left and right lane pixels
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one-by-one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If we found more than minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second-order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Plot the image
    if saveFile:
        # Generate x and y values
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Combine the images
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Plot
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='blue')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(filename)

    # Return
    return left_fit, right_fit


def margin_search(img, left_fit, right_fit, margin=100, saveFile=0, filename=''):
    """
    After a successful sliding window search, just search in predefined
    margins around the previous line position

    img: The image
    left_fit: Polynomial fit of the left lane line in the previous image
    right_fit: Polynomial fit of the right lane line in the previous image
    margin: The width of windows +/- this margin
    saveFile: Whether to save an output figure
    filename: The name of the file if saving an output figure
    return: (left_fit) The left polynomial, (right_fit) The right polynomial
    """
    # Create an output image
    out_img = np.dstack((img, img, img)) * 255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Generate indices of left and right lane pixels
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second-order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Plot the image
    if saveFile:
        # Generate x and y values
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Combine the images
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Plot
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='blue')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(filename)

    # Return
    return left_fit, right_fit


def radius_of_curvature(img, left_fit, right_fit):
    """
    Determines the radius of curvature, in meters, of the left and right lanes

    img: The image 
    left_fit: The left polynomial fit
    right_fit: The right polynomial fit
    return (left_curverad) The left line radius of curvature in meters,
            (right_curverad) The right line radius of curvature in meters
            (avg_curverad) The average of the left and right curves
    """
    # Get the radius in pixel space
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Define meters to pixel conversion
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    # Generate x and y values
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Calculate the average
    avg_curverad = np.mean([left_curverad, right_curverad])

    # Return the radius of curvature in meters
    return left_curverad, right_curverad, avg_curverad


def distance_from_center(left_fit, right_fit):
    """
    Calculates the distance from the center

    :param left_fit: The left polynomial fit
    :param right_fit: The right polynomial fit
    :return: (str) The string descriptor e.g. "1.00m left",
             (left) True if left, false if right
             (meters) The distance left or right in meters
    """

    y = 700
    mid_x = 650
    xm_per_pix = 3.7 / 700

    x_left = left_fit[0] * (y**2) + left_fit[1] * y + left_fit[2]
    x_right = right_fit[0] * (y**2) + right_fit[1] * y + right_fit[2]
    meters = xm_per_pix * ((x_left + x_right) / 2 - mid_x)

    left = meters < 0
    meters = np.absolute(meters)

    str = "%.2fm %s" % (meters, 'left' if left else 'right')

    return str, left, meters


def draw_lines(undist, warped, left_fit, right_fit, Minv):
    """
    Draws fit lane lines back on the original image

    undist: The original (undistorted) image
    warped: The warped image
    left_fit: The polynomial fit for the left lane line
    right_fit: The polynomial fit for the right lane line
    Minv: The inverse perspective matrix
    return The undistorted image with lane lines drawn
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into a usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to the original image space
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image and return
    return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


"""
2. Threshold Calculations
"""


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
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
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale_factor = np.max(mag_sobel) / 255
    mag_scaled = (mag_sobel / scale_factor).astype(np.uint8)

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
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

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
    S = hlsimg[:, :, 2]

    # Create a mask with the threshold applies
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # Return the mask
    return binary


# def threshold(undist):
#     """ 
    
#     """
#     # Copy the image
#     img = np.copy(undist)

#     # Convert to HLS
#     hlsimg = hls(img)

#     # Define the white color mask
#     white_min = np.array([0, 210, 0], dtype=np.uint8)
#     white_max = np.array([255, 255, 255], dtype=np.uint8)
#     mask_white = cv2.inRange(hlsimg, white_min, white_max)

#     # Define the yellow color mask
#     yellow_min = np.array([18, 0, 100], dtype=np.uint8)
#     yellow_max = np.array([30, 220, 255], dtype=np.uint8)
#     mask_yellow = cv2.inRange(hlsimg, yellow_min, yellow_max)

#     # Apply magnitude, direction, and saturation thresholds

#     # Generate the combined function
#     combined = np.zeros_like(mask_white)



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
    out = []  # Output images

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... (7,5,0)
    objp = np.zeros((ydim * xdim, 3), np.float32)
    objp[:, :2] = np.mgrid[0:xdim, 0:ydim].T.reshape(-1, 2)  # x, y coordinates

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

def fit_mvg_avg(left_fit, right_fit, num=5, diff=1e-3):
    """
    """
    global left_fit_prev, right_fit_prev

    left_fit_hist.append(left_fit)
    right_fit_hist.append(right_fit)

    # Determine whether this is the first frame in the video
    leftHistLen = len(left_fit_hist)
    if leftHistLen > 1:
        # Truncate history if it is longer than the moving average number
        if leftHistLen > num:
            left_fit_hist.pop(0)
        if len(right_fit_hist) > num:
            right_fit_hist.pop(0)

        # Get the moving average for each fit polynomial
        left_fit_avg = np.average(left_fit_hist, axis=0)
        right_fit_avg = np.average(right_fit_hist, axis=0)

        # if np.sign(left_fit_avg[0]) == np.sign(right_fit_avg[0]):
        if np.absolute(left_fit_avg[0] - left_fit_prev[0])  <= diff:
            left_fit_prev = left_fit_avg
        if np.absolute(right_fit_avg[0] - right_fit_prev[0])  <= diff:
            right_fit_prev = right_fit_avg
    else:
        # Set the previous value to the present fit value
        left_fit_prev = left_fit
        right_fit_prev = right_fit

    # Return the values
    return left_fit_prev, right_fit_prev   


def img_process_pipeline(img, ksize=3, saveFile=0, fname='', smoothing=1):
    """
    The pipeline for image processing

    img: The image to process
    ksize: The kernel size
    src: The source array of detected corners from the calibration step
    saveFile: Boolean: If true, saves the image at various steps along
              the pipeline
    fname: Optional filename parameter, req if saveFile is True
    smoothing: Turn smoothing (moving average) on or off 
    return (new_img) The processed image
    """
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



    # Apply each of the thresholding functions
    # gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100)) # (20, 100), (230,255)
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100)) # (20, 100)
    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100)) #old (30, 100)
    # dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))  # (0.7,1.3)
    # sat_binary = saturation_thresh(undist, thresh=(200, 255))  # old (200, 255), (180, 255)
    # gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(120, 254))  # (20, 100), (230,255)
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(0, 255))  # (20, 100)
    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(245, 254))  # old (30, 100), (220,255)
    # dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(1.8, 2.5))  # (0.7,1.3)
    # sat_binary = saturation_thresh(undist, thresh=(180, 255))  # old (200, 255), (180, 255)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(120, 255)) # (20, 100), (230,255)
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100)) # (20, 100)
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(30, 100)) #old (30, 100)
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))  # (0.7,1.3)
    sat_binary = saturation_thresh(undist, thresh=(200, 255))  # old (200, 255), (180, 255)

    # 17-05-11 22:46 changed dir binary to 1.8, 2.5
    # 17-05-12 06:57 Changed gradx to 120, 255, dir binary back to 0.7, 1.3
    # 17-05-12 07:52 changed mag_binary to 230, 254
    # 17-05-12 08:23 changed mag_binary back to 30, 100

    # Combine the thresholding results
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sat_binary == 1)] = 1
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1)) | (sat_binary == 1)] = 1

    if saveFile:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist)
        ax1.set_title('Undistorted Image', fontsize=50)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_img_dir + '/combined_' + fname.split('/')[-1])

    # Perspective transform (bird's eye view )
    warped, M, Minv = perspective_transform(combined)
    if saveFile:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist)
        ax1.set_title('Undistorted Image', fontsize=50)
        ax2.imshow(warped, cmap='gray')
        ax2.set_title('Undist. & Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(out_img_dir + '/persp_' + fname.split('/')[-1])

    # Get the histogram of the bottom half of the image
    hist = histogram(warped)
    if saveFile:
        plt.figure()
        plt.plot(hist)
        plt.savefig(out_img_dir + '/hist_' + fname.split('/')[-1])

    # Initialize the polynomial fit values for each line
    left_fit = []
    right_fit = []

    # Determine if this is the first image
    if len(left_fit_prev) == 0:
        # Perform a sliding window search to get the polynomial fit of each line
        if saveFile:
            left_fit, right_fit = sliding_window(warped, hist, saveFile=1,
                                                filename=out_img_dir + '/slide_' +                                                       fname.split('/')[-1])
        else:
            left_fit, right_fit = sliding_window(warped, hist)
    else: 
        # Perform a margin search with the previous fit values
        left_fit, right_fit = margin_search(warped, left_fit_prev, right_fit_prev, margin=40)

    # Apply smoothing
    if smoothing:
        left_fit, right_fit = fit_mvg_avg(left_fit, right_fit, num=3, diff=4e-4)

    # Get the radius of curvature for both lines
    left_curverad, right_curverad, avg_curverad = radius_of_curvature(warped, left_fit, right_fit)
    if saveFile:
        print('left:', left_curverad, 'm, right:', right_curverad, 'm, avg:', avg_curverad, 'm')

    # Print the radius of curvature on the undistorted image
    cv2.putText(undist, 'Radius of Curvature: %.2fm' % avg_curverad, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # Get the distance from the center
    str, left, meters = distance_from_center(left_fit, right_fit)

    # Print the distance from center on the undistorted image
    cv2.putText(undist, "Dist. from Center: " + str, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Plot the calculated lane lines on the undistorted image
    new_img = draw_lines(undist, warped, left_fit, right_fit, Minv)
    if saveFile:
        plt.figure()
        plt.imshow(new_img)
        plt.savefig(out_img_dir + '/replot_' + fname.split('/')[-1])

    # Return the augmented image
    return new_img


def process_video(inFileName, outFileName):
    """
    Run the pipeline on a video

    :inFileName: The video file to process
    :outFileName: The file to store the output video
    :return:
    """
    movie = VideoFileClip(inFileName)
    movie_processed = movie.fl_image(img_process_pipeline)
    movie_processed.write_videofile(outFileName, audio=False)



def test():
    """
    The main project pipeline applied to one test image
    """
    # Calibrate the camera lens using chessboard images
    calibrate_chessboard()

    # Process single images through the pipeline
    for i in range(1, 7):
        fname = test_img_dir + "/test" + str(i) + ".jpg"
        img = mpimg.imread(fname)
        img_process_pipeline(img, ksize=k_size, saveFile=1, fname=fname, smoothing=0)


def main(fileName):
    """
    The main project pipeline

    :fileName: The video to process
    """
    # Calibrate the camera lens using chessboard images
    calibrate_chessboard()

    # Process the video
    process_video(fileName, out_img_dir + '/out_' + fileName.split('/')[-1])


# test()
main('./project_video.mp4')
# main('./challenge_video.mp4')