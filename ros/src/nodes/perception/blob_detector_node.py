#!/usr/bin/env python

from curses import raw
import rospy
import cv2
import numpy as np
import message_filters
import numpy as np
import copy
import tf.transformations
# from PIL import Image
# from PIL import ImageDraw
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, TransformStamped, Twist
from sensor_msgs.msg import CameraInfo, Image as ImageSensor_msg

from blob_detector import BlobDetector

class BlobDetectorNode():
    def __init__(
            self,
    ):
        self.cv_bridge = CvBridge()
        self.publish_freq = rospy.get_param('~publish_freq', 30)
        self.dt = float(1.0 / self.publish_freq)
        self.rate = rospy.Rate(self.publish_freq)
        self.camera_freq = rospy.get_param('~cam_publish_freq', 30)
        self.cam_dt = float(1.0 / self.camera_freq)

        self.detector_params = rospy.get_param('~detector_params')
        
        self.image_msg = None
        self.img = None
        self.camera_info = None
        self.depth_image_msg = None
        self.depth_image = None
        self.new_image_received = False
        
        self.tstep = 0
        self.start_t = 0

        self.detector = BlobDetector(
            **self.detector_params
        )

        self.pub_detection = \
            rospy.Publisher(
                '~detected_img',
                ImageSensor_msg,
                queue_size=1)

        self.pub_mask = \
            rospy.Publisher(
                '~mask_img',
                ImageSensor_msg,
                queue_size=1
            )

        self.pub_pose = \
            rospy.Publisher(
                '~obj_pose',
                PoseStamped,
                queue_size=1
            )
        
        self.pub_pose_filtered = \
            rospy.Publisher(
                '~obj_pose_filtered',
                PoseStamped,
                queue_size=1
            )

        self.pub_camera_info = \
            rospy.Publisher(
                '~camera_info',
                CameraInfo,
                queue_size=10
            )

        # Start ROS subscriber
        image_sub = message_filters.Subscriber(
            rospy.get_param('~topic_camera', '/camera/color/image_raw'),
            ImageSensor_msg
        )
        info_sub = message_filters.Subscriber(
            rospy.get_param('~topic_camera_info', '/camera/color/camera_info'),
            CameraInfo
        )
        # pointcloud_sub = message_filters.Subscriber(
        #     rospy.get_param('~topic_pointcloud'),
        #     PointCloud2
        # )
        depth_image_sub = message_filters.Subscriber(
            rospy.get_param('~topic_depth_image', '/camera/depth/image_rect_raw'),
            ImageSensor_msg
        )

        ts = message_filters.TimeSynchronizer([image_sub, depth_image_sub, info_sub], 1)
        ts.registerCallback(self.image_callback)


    def image_callback(self, image_msg, depth_image_msg, camera_info):
        """Image callback"""

        self.image_msg = image_msg
        self.img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        self.camera_info = camera_info
        self.depth_image_msg = depth_image_msg
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg, depth_image_msg.encoding)
        self.new_image_received = True

    def publish_loop(self):
        
        while not rospy.is_shutdown():
            try:
                if self.img is not None and self.depth_image is not None and self.camera_info is not None:
                    detections, im_with_keypoints, masked_im = self.detector.get_detections(
                        self.img, self.depth_image, self.camera_info)

                    print(detections)

                    im_with_keypoints_msg = self.cv_bridge.cv2_to_imgmsg(im_with_keypoints, encoding="passthrough")
                    im_with_keypoints_msg.header = self.image_msg.header
                    im_with_keypoints_msg.header.stamp = rospy.Time.now() 
                    # im_with_keypoints_msg.header.frame_id =  self.image_msg.header.frame_id
                    self.pub_detection.publish(im_with_keypoints_msg)
                    self.pub_mask.publish(self.cv_bridge.cv2_to_imgmsg(masked_im, encoding="passthrough"))
                    self.pub_camera_info.publish(self.camera_info)

                    self.new_image_received = False

                if self.tstep == 0:
                    self.start_t = rospy.get_time()
                self.tstep = rospy.get_time() - self.start_t
                self.rate.sleep()

            except rospy.exceptions.ROSTimeMovedBackwardsException:
                print('time moved backwards')
                continue


    def close(self):
        print('Closing node')
        self.pub_detection.unregister()
        self.pub_mask.unregister()
        self.pub_pose.unregister()
        self.pub_pose_filtered.unregister()
        self.pub_camera_info.unregister()

def main():
    """Main routine to run blob detector"""

    # Initialize ROS node
    rospy.init_node('blob_detector')
    node = BlobDetectorNode()

    rospy.loginfo("Running Blob Detector...  (Listening to camera topic: '{}')".format(
        rospy.get_param('~topic_camera')))
    rospy.loginfo("Ctrl-C to stop")

    try:
        node.publish_loop()
    except rospy.ROSInterruptException:
     # rospy.ROSInterruptException:
        rospy.loginfo('[BlobDetector]: Got RosInterruptException')
        node.close()

if __name__ == "__main__":
    main()