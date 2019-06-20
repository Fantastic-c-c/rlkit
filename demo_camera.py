import cv2
import datetime

video_path = 'demo/video_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".avi"
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 1080)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
