import rospy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from utils.box_op import Box

class DetectionPublisher:
    def __init__(self):
        self.pub = rospy.Publisher("detector", String)
        rospy.init_node("bbox_gen", anonymous=True)
        # self.rate = rospy.Rate(10)
    
    def send_boxes(self, bboxes):
        for b in bboxes:
            rospy.loginfo(b)
            self.pub.publish(b)


class CameraSubscriber:
    def __init__(self):
        rospy.init_node("img_listener")

        """ Create the cv_bridge object """
        self.bridge = CvBridge()

        self.cv_image = None

        """ Subscribe to the raw camera image topic """
        self.sub = rospy.Subscriber("img_raw", Image, self.callback)

        self.stopped = False

        rospy.spin()

    def callback(self, data):
        try:
            """ Convert the raw image to OpenCV format """
            self.cv_image = self.bridge.imgmsg_to_cv(data, "bgr8")

        except CvBridgeError:
            print(CvBridgeError) 
    
    def get_frame(self):
        return self.cv_image

    def is_running(self):
        if self.stopped:
            return False
        else:
            return True