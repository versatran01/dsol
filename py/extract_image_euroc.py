import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# EuRoC MAV Dataset calib.txt: 435.2046959714599 435.2046959714599 367.4517211914062 252.2008514404297 0.11007784219

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':
    bag_file = "/home/username/dataset/dsol/V1_03_difficult.bag"
    bag_name = "v103"
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    topics = [
        "/cam0/image_raw",
        "/cam1/image_raw"
    ]

    data_dir = Path("/home/username/dataset/dsol/realsense") / bag_name
    left_dir = data_dir / "infra1"
    right_dir = data_dir / "infra2"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    

    K1 = np.array([[458.654, 0.0, 367.215],[0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
    D1 = np.array([[-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0]])
    R1 = np.array([[0.999966347530033, -0.001422739138722922, 0.008079580483432283],[0.001365741834644127, 0.9999741760894847, 0.007055629199258132], [-0.008089410156878961, -0.007044357138835809, 0.9999424675829176]])
    P1 = np.array([[435.2046959714599, 0, 367.4517211914062, 0],[0, 435.2046959714599, 252.2008514404297, 0], [0, 0, 1, 0]])
    
    K2 = np.array([[457.587, 0.0, 379.999],[0.0, 456.134, 255.238], [0.0, 0.0, 1.0]])
    D2 = np.array([[-0.28368365, 0.07451284, -0.00010473, -3.555907e-05, 0.0]])
    R2 = np.array([[0.9999633526194376, -0.003625811871560086, 0.007755443660172947],[0.003680398547259526, 0.9999684752771629, -0.007035845251224894], [-0.007729688520722713, 0.007064130529506649, 0.999945173484644]])
    P2 = np.array([[435.2046959714599, 0, 367.4517211914062, -47.90639384423901],[0, 435.2046959714599, 252.2008514404297, 0], [0, 0, 1, 0]])
    
    
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (752, 480), cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (752, 480), cv2.CV_32FC1)

    for topic, msg, t in tqdm(bag.read_messages(topics=topics)):
        count = t.to_nsec() // 1000000

        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if topic == topics[0]:
            left_rectified = cv2.remap(cv_img, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(left_dir / f"image{count:08d}.png"), left_rectified)
        elif topic == topics[1]:
            right_rectified = cv2.remap(cv_img, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(right_dir / f"image{count:08d}.png"), right_rectified)
    bag.close()
