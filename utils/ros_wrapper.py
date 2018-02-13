import cv2
import rospy

from cv_bridge import CvBridge, CvBridgeError
from threading import Thread
from utils.fps import FPS

class CameraPublisher:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise Exception("Video/Camera device not found at: {}".format(src))
        
        self.pub = rospy.Publisher("camera", String)
        rospy.init_node("img_raw", anonymous=True)

        (self.grabbed, self.frame) = self.stream.read()

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
            self.pub.publish(self.frame)
            self.f.update()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.f.stop()
    
    def get_dimensions(self):

        c = int(self.stream.get(3))  
        r = int(self.stream.get(4)) 
        return r, c
    
    def get_raw_frames(self):
        return self.f.get_frames()
