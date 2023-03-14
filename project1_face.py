#!/usr/bin/env python
from __future__ import print_function

# import sys
import rospy
import cv2
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from qt_nuitrack_app.msg import Faces, FaceInfo

import face_recognition
import os
import numpy as np


def LoadEncodings(dir):
    faces=os.listdir(dir)
    images_known = []
    for x in faces:
        images_known.append(dir+"/"+x)
    known_face_encodings = []
    known_face_names = []
    for x in images_known:
        known_image = face_recognition.load_image_file(x)
        known_face_encoding = face_recognition.face_encodings(known_image,model="small", num_jitters=1)[0]
        known_face_encodings.append(known_face_encoding)
        known_face_names.append(os.path.basename(x))

    return known_face_encodings,known_face_names


def SingleFaceRecognition( image,known_face_encodings,known_face_names):
    unknown_image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(unknown_image,model="hog")
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations,model="small", num_jitters=1)
    name=""
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            name=name.split("_")[0]
    return name

class image_converter:
    faces = None
    faces_time = None

    def __init__(self):
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/project1_face/out", Image, queue_size=1)
        print(self.image_pub)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.image_callback)
        self.face_sub = rospy.Subscriber("/qt_nuitrack_app/faces", Faces, self.face_callback)


    def face_callback(self, data):
        self.lock.acquire()
        self.faces = data.faces
        self.faces_time = rospy.Time.now()
        self.lock.release()

    def image_callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        self.lock.acquire()
        new_faces = self.faces
        new_faces_time = self.faces_time
        self.lock.release()
        
        if new_faces and (rospy.Time.now()-new_faces_time) < rospy.Duration(5.0):
            for face in new_faces:
                rect = face.rectangle
                face_pred=SingleFaceRecognition(cv_image,known_face_encodings1,known_face_names1)
                cv2.rectangle(cv_image, (int(rect[0]*cols),int(rect[1]*rows)),
                                      (int(rect[0]*cols+rect[2]*cols), int(rect[1]*rows+rect[3]*rows)), (0,255,0), 2)
                x = int(rect[0]*cols)
                y = int(rect[1]*rows)
                w = int(rect[2]*cols)
                h = int(rect[3]*rows)
        
                cv2.putText(cv_image, face_pred, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, lineType=cv2.LINE_AA)

             
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)




if __name__ == '__main__':
    rospy.init_node('project1_face', anonymous=True)
    known_face_encodings1,known_face_names1=LoadEncodings("/home/qtrobot/catkin_ws/src/project1_face/src/Students")
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
