import cv2
import numpy as np

count = 0

# obtains the image from the camera
video_cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frames = video_cap.read()


    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        cv2.imwrite("nsei"+str(count)+".jpg", frames)
        count += 1

    # press ESC to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_cap.release()
cv2.destroyAllWindows()