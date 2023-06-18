import cv2
import copy
import numpy as np

class BlobDetector():
    def __init__(self):
        self.params = cv2.SimpleBlobDetector_Params()
        self.h_nominal = 85.0
        self.hsv_thresh = [10, 20, 10]
        self.num_largest_detections = 1

        # self.minHSV = np.array([self.h_nominal - self.hsv_thresh[0], 100-self.hsv_thresh[1], 100-self.hsv_thresh[2]])
        # self.maxHSV = np.array([self.h_nominal + self.hsv_thresh[0], 255, 255])
        self.minHSV = np.array([0, 0, 0])
        self.maxHSV = np.array([255, 255, 255])


        #Change thresholds
        self.params.minThreshold = 40
        self.params.maxThreshold = 255

        self.params.minDistBetweenBlobs = 300
        
        self.params.filterByColor = False

        self.params.filterByArea = True
        self.params.minArea = 500
        self.params.maxArea = 5000

        # Filter by Circularity
        self.params.filterByCircularity = False
        self.params.minCircularity = 0.1

        # Filter by Convexity
        self.params.filterByConvexity = False
        self.params.minConvexity = 0.87

        # Filter by Inertia
        self.params.filterByInertia = False
        self.params.minInertiaRatio = 0.01

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv2.SimpleBlobDetector(self.params)
        else : 
            self.detector = cv2.SimpleBlobDetector_create(self.params)


    def get_detections(self, img, depth_img=None):
        return self.detect_keypoints(img, depth_img)

    def detect_keypoints(self, img, depth_img=None):
        detections = {}
        #gaussian blur
        blurred_img = cv2.GaussianBlur(img,(7,7),0) 
        # blurred_depth_img = cv2.GaussianBlur(self.depth_image,(7,7),0)
        #convert to hsv
        hsv_image = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        # gray_image = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
        # #color filter
        # all_keypoints = []
        # for c in range(self.num_colors):
        mask_hsv = cv2.inRange(hsv_image, self.minHSV, self.maxHSV) 
        print(np.min(hsv_image), np.max(hsv_image), self.minHSV, self.maxHSV)
        masked_rgb = cv2.bitwise_and(img, img, mask = mask_hsv)
        masked_gray = (cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)) 
        # Detect blobs.
        curr_keypoints = self.detector.detect(masked_gray)
            # if c == 0:
            #     blended_masked_rgb = copy.deepcopy(masked_rgb)
            #     blended_masked_gray = copy.deepcopy(masked_gray)
            # else:
            #     blended_masked_rgb = cv2.add(blended_masked_rgb, masked_rgb)
            #     blended_masked_gray = cv2.add(blended_masked_gray, masked_gray)

        sorted_keypoints = sorted(curr_keypoints, key=lambda x: x.size, reverse=True)
        # (keypoints, descriptors) = self.orb.compute(masked_gray, sorted_keypoints)
        # all_keypoints += sorted_keypoints
        if len(sorted_keypoints) > 0:
            for i in range(min(len(sorted_keypoints), self.num_largest_detections)):
                # detection_key = self.blob_frame_id_base[c] + "/" + str(i)
                x_cam = sorted_keypoints[i].pt[0]
                y_cam = sorted_keypoints[i].pt[1]
                
                #Get depth value
                # z_patch = blurred_depth_img[int(y_cam)-3:int(y_cam)+3, int(x_cam)-3:int(x_cam)+3] #depth in mm for realsense
                # z_patch = self.depth_image[int(y_cam)-3:int(y_cam)+3, int(x_cam)-3:int(x_cam)+3] #depth in mm for realsense
                # #remove zero depths
                # z_patch = z_patch[z_patch > 0]
                
                # if len(z_patch) == 0 or np.sum(np.isnan(z_patch)):
                #     rospy.loginfo('Got bad z, skipping \n'.format(detection_key))
                #     continue

                # z = np.average(z_patch)
                z = 0.0
                if depth_img is not None:
                    z = depth_img[int(y_cam), int(x_cam)]
                    z /= 1000.0

                    P = np.array(self.camera_info.P).reshape(3,4)
                    cam_coords = np.linalg.pinv(P) @ np.array([x_cam,y_cam,1.0])
                    x = z * cam_coords[0]; y = z * cam_coords[1]
                    curr_pos = np.array([x,y,z])
                else:
                    curr_pos = np.array([x_cam, y_cam])
                # if (self.last_detections is None) or (detection_key not in self.last_detections):
                #     curr_vel = np.zeros(3)
                # else:
                    # last_pos = self.last_detections[detection_key][0:3]
                    # curr_vel = (curr_pos - last_pos) / self.cam_dt
                curr_vel = np.zeros(2)
                curr_state = np.concatenate((curr_pos, curr_vel))
                # loc = np.array([[x,y,z]]).T
                #find closest keypoint in last frame
                # if len(self.last_sorted_keypoints) > 0:
                #     # print('comparing')
                    # closest_keypoint_idx = (np.linalg.norm(self.last_keypoint_locs - loc, axis=0)).argmin()
                    # print(np.linalg.norm(self.last_keypoint_locs - loc, axis=0))
                    # brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
                    # no_of_matches = brute_force.match(descriptors,self.last_descriptors)
                
                    # finding the humming distance of the matches and sorting them
                    # no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
                    # print(no_of_matches[0].distance)
                    # print(i, closest_keypoint_idx, loc, self.last_keypoint_locs)
                # else:
                    # closest_keypoint_idx = i
                # closest_keypoint_idx = i
                # temp_locs[:,closest_keypoint_idx] = loc[:,0] 
                detections = curr_state
        
        self.last_detections = copy.deepcopy(detections)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(masked_gray, curr_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # im_with_keypoints_msg = self.cv_bridge.cv2_to_imgmsg(im_with_keypoints, encoding="passthrough")
        # im_with_keypoints_msg.header = self.image_msg.header
        # im_with_keypoints_msg.header.stamp = rospy.Time.now() 
        # # im_with_keypoints_msg.header.frame_id =  self.image_msg.header.frame_id
        # self.pub_detection.publish(im_with_keypoints_msg)
        # self.pub_mask.publish(self.cv_bridge.cv2_to_imgmsg(blended_masked_rgb, encoding="passthrough"))
        # self.pub_camera_info.publish(self.camera_info)
        return detections, im_with_keypoints, mask_hsv, masked_rgb
    




if __name__ == "__main__":
    img = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)
    detector = BlobDetector()

    detections, im_with_keypoints, hsv_image, masked_image = detector.get_detections(img)
    print(detections)
    cv2.namedWindow('InputImage', cv2.WINDOW_NORMAL)
    cv2.imshow('InputImage', img)

    cv2.namedWindow('KeypointImage', cv2.WINDOW_NORMAL)
    cv2.imshow('KeypointImage', im_with_keypoints)

    cv2.namedWindow('MaskedImage', cv2.WINDOW_NORMAL)
    cv2.imshow('MaskedImage', hsv_image)

    cv2.waitKey(0)