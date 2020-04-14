import cv2
import numpy as np


BACKGROUND = None
ACCUMULATED_WEIGHT = 0.5

# ROI for detecting the hand
roi_top = 40
roi_bottom = 340
roi_right = 300
roi_left = 600

# Compute Weighted average of image
def calc_accum_avg(frame, accumulated_weight):
    global BACKGROUND
    
    # For first run
    if BACKGROUND is None:
        BACKGROUND = frame.copy().astype("float")
        return None

    # Compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, BACKGROUND, ACCUMULATED_WEIGHT)

# Segmenting the image
def segment(frame, threshold=25):
    global BACKGROUND
    
    diff = cv2.absdiff(BACKGROUND.astype("uint8"), frame)

    # Apply a threshold to the image 
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours form the image
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        # The largest contour will be the hand
        hand_segment = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment)

# Using the webcam to capture and save the images

cam = cv2.VideoCapture(0)

num_frames = 0
i = 420
name = '.png'

while True:
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # Grab the ROI from the frame
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 30 frames we will calculate the average of the background.
    # We will tell the user while this is happening
    if num_frames < 60:
        calc_accum_avg(gray, ACCUMULATED_WEIGHT)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        # Segment the hand region
        hand = segment(gray)
        if hand is not None:
            
            # Unpack
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

            # Display the thresholded image
            cv2.imshow("Thesholded", thresholded)
            
            

    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

    num_frames += 1
    
    # Display the frame with segmented hand
    cv2.imshow("Gesture Capture", frame_copy)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break
    if k == ord('q'):
        # Save the thresholded image in dataset
        # Change folder name here to get different datasets
        # Press q to capture image
        # Press Esc to quit
        cv2.imwrite("data/fist/"+str(i)+name, thresholded) #change folder name here <-
        i += 1


cam.release()
cv2.destroyAllWindows()