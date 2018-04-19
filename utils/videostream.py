# Modified from webcamvideostream.py under https://github.com/jrosebr1/imutils 
from threading import Thread
import cv2
from utils.fps import FPS

class WebcamVideoStream:
    def __init__(self, src=0, res=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception("Video/Camera device not found at: {}".format(src))

        (self.grabbed, self.frame) = self.stream.read()

        self.resize = None
        res = (480, 480)
        # res is a tuple of (width, height)
        if res is not None:
            self.resize = res

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        self.f = FPS()
        self.f.start()

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.f.update()

    def read(self):
        # return the frame most recently read
        if self.resize is not None:
            self.frame = cv2.resize(self.frame, self.resize)
        return self.grabbed, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.f.stop()
        # TODO: Weird error "VIDIOC_DQBUF: Invalid argument"
        self.stream.release()
    
    def get_dimensions(self):
        if self.resize is not None:
            return self.resize[0], self.resize[1]
            
        c = int(self.stream.get(3))  
        r = int(self.stream.get(4)) 
        return r, c
    
    def get_raw_frames(self):
        return self.f.get_frames()
    
    def is_running(self):
        if self.stopped:
            return False
        else:
            return True
