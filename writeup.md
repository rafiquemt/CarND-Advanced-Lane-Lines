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
[threshold_image]: ./output_images/threshold_test_images/test3.jpg "Binary Example"
[warped_image]: ./writeup/warped_image.png "Warp Example"
[fitted_lanes]: ./writeup/fitted_lanes.png "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained `p4.py` in the `calibrate_camera` method on line 206.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][camera_calibrate]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
With the calibration parameters `mtx, dist` computed in the calibration stage, the raw camera images are corrected for distortion using the `cv2.undistort` method. Example below shows an undistorted image from the camera
![alt text][road_transformed]

Note that this image isn't from the input project video, but it is from the `test_images` set and it makes it very easy to see that the distortion correction works because the signs mounted above are correctly straightened out.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image.  The color space was converted into HLS. The S channel was used for color thresholding. The L channel was used for x-gradient magnitude thresholding. The code can be found in the `color_sobel_threshold` method in `p4.py` on line `286`.

![alt text][threshold_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The first step is to calculate the warping parameters for the perspective transform. This is done in `p4.py` in the `get_warp_params` method on line 245. The method uses a hardcoded set of points from a camera image. The `src` points are in the normal image and the `dst` points are intended to be transformed so that we get the perspective of looking top down at the road. The points are passed on to `cv2.getPerspectiveTransform` to get the transformation. It also calculates the inverse perspective transform to go from warped to normal points so that I could plot the lines on the undistorted image.

```
# coordinates of road in normal image
src = np.float32([[184, 645], [583, 445], [697, 445], [1111, 645]])
# transferring to:
dst = np.float32([[150, 700], [150, 20], [1130, 20], [1130, 700]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 184, 645      | 150, 700      |
| 583, 445      | 150, 20       |
| 697, 445      | 1130, 20      |
| 1111, 645     | 1130, 700     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_image]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I split the image vertically into 9 windows of width 50 and started from the bottom. The starting point for the bottom window was determined by taking histogram in the lower half of the rectified and thresholded binary image to find x coordinates where the pixel density is highest. The x-position of the next window moving up is taken as the mean x-coordinate of non-zero pixels if a certain critical threshold is met. This helps slide the window towards the the center of the lane as it curbes

This process of sliding windows is done for the left and right halves of the images so that we can find the left and right lanes.

The code can be found in `p4.py` in the `sliding_window_histogram_new` method on line 50. An image showing the resulting lines found and the fitted polynomial is plotted below

![alt text][fitted_lanes]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

