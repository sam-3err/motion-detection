import cv2
import time

video = cv2.VideoCapture(0)

first_frame = None

while True:
    check, frame = video.read()  # reading data extracted by video capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting to gray for smoother detection
    gray = cv2.GaussianBlur(gray, (11,11), 0)  # blurring to reduce noise

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame, gray)  # difference between frames
    threshold_frame = cv2.adaptiveThreshold(delta_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold_frame = cv2.erode(threshold_frame, None, iterations=1)  # Erosion to remove small white noise
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=1)  # dilate to fill in gaps

    # Find contours of the motion areas
    (cntr, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through the contours
    for contour in cntr:
        if cv2.contourArea(contour) < 1000:  # Ignore small contours
            continue
        
        # Get bounding box coordinates for detected motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)  # draw rectangle around moving object

    # Display the result
    cv2.imshow("motion_tv", frame)

    # Check if 'q' is pressed to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
# Release resources and close any open windows
video.release()
cv2.destroyAllWindows()
