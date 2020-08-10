#!/usr/bin/env python3

import numpy as np
import cv2
import roslib
import rospy
import struct
import math
import time
import argparse
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os 
import message_filters

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor

from lib.config import add_hoircnn_default_config
from lib.predictor import VisualizationDemo

#### args
def get_parser():
	parser = argparse.ArgumentParser(description="Demo for builtin models")

class hoi_prediction(object):
	def __init__(self):
		self.is_compressed = False
		r = rospkg.RosPack()
		self.path = r.get_path('zero-shot-hoi')
		self.cv_bridge = CvBridge() 
		
		#### cfg
		self.args = get_parser()
		self.cfg = get_cfg()
		add_hoircnn_default_config(self.cfg)
		self.cfg.MODEL.WEIGHTS = os.path.join(self.path, "weights", "hico_det_pretrained.pkl")
		
		self.cfg.merge_from_file(os.path.join(self.path, "src", "configs/HICO-DET/interaction_R_50_FPN.yaml"))

		# self.cfg.merge_from_file("./configs/HICO-DET/interaction_R_50_FPN.yaml")
		# self.cfg.merge_from_list(" ")

		#### Set score_threshold for builtin models
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
		self.cfg.MODEL.ROI_HEADS.HOI_SCORE_THRESH_TEST = 0.001

		self.pretictor = VisualizationDemo(self.cfg, self.args)

		#### Publisher
		self.predict_img_pub = rospy.Publisher("prediction_img", Image, queue_size = 1)
		
		#### msg filter 
		# image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		# depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		# ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		# ts.registerCallback(self.callback)
		image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

		print("Finish load model.")

	def callback(self, img_msg):
		try:
			cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
		except CvBridgeError as e:
			print(e)

		predictions, visualized_output = self.pretictor.run_on_image(cv_image)

		self.predict_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(visualized_output.get_image()[:, :, ::-1], "bgr8"))
		print("Detected 1 frame !!!")

	def onShutdown(self):
		rospy.loginfo("Shutdown.")	



if __name__ == '__main__': 
	rospy.init_node('hoi_prediction',anonymous=False)
	hoi_prediction = hoi_prediction()
	rospy.on_shutdown(hoi_prediction.onShutdown)
	rospy.spin()
