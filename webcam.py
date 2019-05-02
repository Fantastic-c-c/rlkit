import cv2
import datetime
import time
from threading import Thread, Event

class Webcam:
    def __init__(self,
                 video_path,
                 img_folder,
                 cap_num=0,
                 width=1920,
                 height=1080):
        self.cap =  cv2.VideoCapture(cap_num)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH , width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT , height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self.fps = fps

        self.writer = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))
        self.thread = Thread(target=self.record)
        self.stop_event = Event()

        self.img_folder = img_folder

    def start_record(self):
        # Flush buffer
        for i in range(int(self.fps)):
            self.cap.grab()

        self.thread.start()

    def take_picture(self):
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()
            img_path = self.img_folder + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".png"
            cv2.imwrite(img_path, frame)
        else:
            print("ERROR taking picture - capture is not opened")

    def record(self):
        while(self.cap.isOpened() and not self.stop_event.is_set()):
            ret, frame = self.cap.read()
            if ret == True:
                # write the flipped frame
                self.writer.write(frame)
                # cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    def stop_record(self):
        self.stop_event.set()
        self.thread.join()

    def close(self):
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
