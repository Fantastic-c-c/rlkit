import cv2
from threading import Thread, Event

class Webcam:
    def __init__(self,
                 video_path,
                 width=1920,
                 height=1080):
        self.cap =  cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH , width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT , height)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self.writer = cv2.VideoWriter(video_path, fourcc, fps, (1920, 1080))
        self.thread = Thread(target=self.record)
        self.stop_event = Event()

    def start_record(self):
        self.thread.start()

    def record(self):
        while(self.cap.isOpened() and not self.stop_event.is_set()):
            ret, frame = self.cap.read()
            if ret == True:
                # write the flipped frame
                self.writer.write(frame)

                cv2.imshow('frame', frame)
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
