## Greg Yeutter

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist1]: ./output_images/out_calibration2.jpg "Undistorted 1"
[undist2]: ./output_images/out_calibration10.jpg "Undistorted 2"
[undist3]: ./output_images/out_calibration16.jpg "Undistorted 3"

[undisttest]: ./output_images/undist_test1.jpg "Undistorted Test"
[combinedtest]: ./output_images/combined_test1.jpg "Combined Test"
[warpedtest]: ./output_images/persp_test1.jpg "Warped Test"
[histtest]: ./output_images/hist_test1.jpg "Histogram Test"
[polyfittest]: ./output_images/slide_test1.jpg "Polynomial Fit Test"
[newimgtest]: ./output_images/replot_test1.jpg "Area Drawn Test"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained under `main.py` in the function `calibrate_chessboard()`, with additional functions `grayscale()` and `undistort()`.  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Under `# Process each calibration image`, each calibration image is imported and converted to grayscale (with `grayscale()`. If corners are detected wirh `cv2.findChessboardCorners()`, the points are added to the objpoints and imgpoints arrays as described in the previous paragraph.

I then call the function `undistort()`. Here, the output `objpoints` and `imgpoints` are to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I apply this distortion correction to each test image using the `cv2.undistort()` function and obtain these results: 

![alt text][undist1]
![alt text][undist2]
![alt text][undist3]

Objectively, most of the corrections appear minor but significant. Most of the distortion appears to be be radial distortion.

### Pipeline (single images)

The pipeline is executed as follows:
* Calibrate the lens using chessboard images

Then, for each image:
* Correct for distortion
* Apply color and gradient thresholds
* Warp the perspective to a bird's-eye view of the lane
* 


#### 1. Provide an example of a distortion-corrected image.

The `undistort()` function is applied to the test image, using the calibration data captured in `calibrate_chessboard()`. An example of this calibration is:

![alt text][undisttest]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps after `# Combine the thresholding results` in `img_process_pipeline()`).  Here's an example of my output for this step:

![alt text][combinedtest]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transform is called within `img_process_pipeline()` and specifically processed in the function `perspective_transform()`. I elected to hard-code the source and destination points as follows, as it was the simplest way to achieve a good result:

```python
# Define the four source points
src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])

# Define the four destination points
dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
```

Using the source and destination points, I calculated the transformation matrix using `cv2.getPerspectiveTransform()`. I then warped the image to a top-down view with `cv2.warpPerspective()`.

In the warped image below, both curved lines appear parallel:

![alt text][warpedtest]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

First, I created a histogram of the bottom half of the warped perspective image. The code is located in the `histogram()` function. This histogram highlights the pixels (from left to right) that are relatively light in color:

![alt text][histtest]

Then, I implemented a sliding window search to get a polynomial fit for each lane line in the image. The polynomial fit of each line is plotted on the warped perspective image for visualization.

![alt text][polyfittest]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

A function, `radius_of_curvature()`, was defined to calculate the radius of curvature of the polynomial fit lines. This function first calculates the radius of curvature in pixel space, then converts it to "real world" space in meters.

For the example image, the radius of curvature was calculated as:

```
left: 261.037646339 m, right: 321.782499731 m
```

This result seems to be a reasonable real-world value based on the U.S. government specifications for highway curvature.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented redrawing onto the undistorted image in the function `draw_lines()`. It draws the area on the warped image, then uses the inverse transform matrix to warp the perspective and recast the area in the plane of the original image. It appears to work well on the test image:

![alt text][newimgtest]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
