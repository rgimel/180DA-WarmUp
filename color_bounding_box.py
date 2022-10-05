import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 0, 180])
    lower_red_hsv = np.array([0, 180, 180])
    upper_red = np.array([80, 80, 255])
    upper_red_hsv = np.array([40, 255, 255])

    mask = cv2.inRange(frame, lower_red, upper_red)
    mask_hsv = cv2.inRange(frame_hsv, lower_red_hsv, upper_red_hsv)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    res_hsv = cv2.bitwise_and(frame_hsv, frame_hsv, mask=mask_hsv)

    cnt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res_hsv = cv2.cvtColor(res_hsv, cv2.COLOR_HSV2BGR)
    cnt_hsv = cv2.cvtColor(res_hsv, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(cnt)
    x_hsv, y_hsv, w_hsv, h_hsv = cv2.boundingRect(cnt_hsv)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(frame_hsv, (x_hsv, y_hsv),
                  (x_hsv+w_hsv, y_hsv+h_hsv), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('frame_hsv', frame_hsv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
