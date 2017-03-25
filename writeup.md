# Advanced Lane Finding Project

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

[camera_calibrate]: ./writeup/calibration.png "Undistorted"
[road_transformed]: ./writeup/camera_undistortion.png "Road Transformed"
[threshold_image]: ./writeup/threshold_test3.jpg "Binary Example"
[warped_image]: ./writeup/warped_image.png "Warp Example"
[fitted_lanes]: ./writeup/fitted_lanes.jpg "Fit Visual"
[final_output]: ./writeup/final-lanes.png "Output"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained `p4.py` in the `calibrate_camera` method on line 230.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][camera_calibrate]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
With the calibration parameters `mtx, dist` computed in the calibration stage, the raw camera images are corrected for distortion using the `cv2.undistort` method. Example below shows an undistorted image from the camera
![alt text][road_transformed]

Note that this image isn't from the input project video, but it is from the `test_images` set and it makes it very easy to see that the distortion correction works because the signs mounted above are correctly straightened out.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.  The color space was converted into HLS. The S channel was used for color thresholding. The L channel was used for x-gradient magnitude thresholding. The code can be found in the `color_sobel_threshold` method in `p4.py` on line `310`.

![alt text][threshold_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The first step is to calculate the warping parameters for the perspective transform. This is done in `p4.py` in the `get_warp_params` method on line 269. The method uses a hardcoded set of points from a camera image. The `src` points are in the normal image and the `dst` points are intended to be transformed so that we get the perspective of looking top down at the road. The points are passed on to `cv2.getPerspectiveTransform` to get the transformation. It also calculates the inverse perspective transform to go from warped to normal points so that I could plot the lines on the undistorted image.

```
# coordinates of road in normal image
src = np.float32([[289, 661], [595, 450], [684, 450], [1017, 661]])
# transferring to:
dst = np.float32([[250, 700], [250, 20], [950, 20], [950, 700]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 289, 661      | 250, 700      |
| 595, 450      | 250, 20       |
| 684, 450      | 950, 20     |
| 1017, 661     | 950, 700     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_image]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I split the image vertically into 9 windows of width 50 and started from the bottom. The starting point for the bottom window was determined by taking histogram in the lower quarter of the rectified and thresholded binary image to find x coordinates where the pixel density is highest. The x-position of the next window moving up is taken as the mean x-coordinate of non-zero pixels if a certain critical threshold is met. This helps slide the window towards the the center of the lane as it curbes

This process of sliding windows is done for the left and right halves of the images so that we can find the left and right lanes.

Finally the points points that are found through the sliding window search are fitted to a polynomial using `np.polyfit`.

The code can be found in `p4.py` in the `sliding_window_histogram_new` method on line 53. An image showing the resulting lines found and the fitted polynomial is plotted below

![alt text][fitted_lanes]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the method `draw_radius_of_curvature_center_location` starting on line 189 in `p4.py`. The curvature is calculated using the formula from the course [notes](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f)

This required fitting the polynomial to points coverted to world space (meters) from image space (pixels). I relied on a lane width of 3.7m which is the minimum width for roads. The lanes are separated by about 700 pixels
The length of the lane is estimate by countding the number of dots. The spacing between the dots on the high ways 14.6m based on the CA highway specs here: [Figure 6.2 detail 13](http://www.dot.ca.gov/trafficops/camutcd/docs/TMChapter6.pdf) on page 6-26.

The offset of the car was calculated by finding the pixel coordinate halfway between the left and right detected lanes. This pixel coordinate was then subtracted from middle of the frame (640px) to get the offset in pixels. The offset in pixels was then coverted to x distance in meters using the conversion factor described above for lane width

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is done on `p4.py` in the method `draw_lines` on line `167`. The left and right lane x coordinates that were generated from the fitted polynomial are plotted using cv2.fillPoly for each y pixel on the warped image where the lanes were detected. The region is filled with green and is stored as a new mask layer. This layer is then perspective transformed using the inverse M matrix to the undistorted image space.

This undistorted mask is now weighted together with the undistorted image to display the lane. The radius of curvature and the offset are also written on the undistorted image as seen below

![alt text][final_output]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

One of the first things was doing the perspective warp. I realized that you don't want to go too far down the lane. The further down you go, more of the lane is compressed in fewer pixels. When you perspective transform that, the pixels are stretched near the top part of the lane. Also make sure that not all the width of the image is used by the width of the lane. If you don't do that, then you miss out curves of the lane.

The thresholding using Sobelx and HLS color space was done through trial and error to find thresholds that captured the lane without introducing too much noise from other shadows or markings in the middle of the lane

Lane finding was done by using a sliding window to find regions with the highest pixel density. It was important to tweak the window width and height to ensure that


** Fail
* Sharp curves
* shadows for extended period of time
* markings close to the lane or road changing

