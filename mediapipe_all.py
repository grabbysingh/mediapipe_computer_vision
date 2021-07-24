# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 15:32:20 2021

@author: grabb
"""

import cv2
import math
import numpy as np
import mediapipe as mp
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
text = '1. Face Detection\n2. Face Mesh\n3. Hand Detection\n4. Holistic Detection\n5. Object Detection\n6. Pose Detection\n7. Selfie Segmentation\n8. Volume Change\n9. Fingers Count\n'
inp = list(map(str, input(text).split()))

if "1" in inp: # 1 face detection
    mp_face_detection = mp.solutions.face_detection
elif "2" in inp: # 2 face mesh detection
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
elif "3" in inp: # 3 hand detection
    mp_hands = mp.solutions.hands
elif "4" in inp: # 4 holistic detection
    mp_holistic = mp.solutions.holistic
elif "5" in inp: # 5 objectron detection
    mp_objectron = mp.solutions.objectron
elif "6" in inp: # 6 pose detection
    mp_pose = mp.solutions.pose
elif "7" in inp: # 7 selfie segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    BG_COLOR = (192, 192, 192) # gray
    bg_image = None
elif "8" in inp:
    mp_hands = mp.solutions.hands
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
elif "9" in inp:
    mp_hands = mp.solutions.hands
    

cap = cv2.VideoCapture(0)
while cap.isOpened():#
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # 1 face detection
    if "1" in inp:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            face_det_results = face_detection.process(image)
    
    # 2 face mesh detection
    if "2" in inp:
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            face_mesh_results = face_mesh.process(image)
    
    # 3 hand detection
    if "3" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            hand_results = hands.process(image)
    
    # 4 holistic detection
    if "4" in inp:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            holistic_results = holistic.process(image)
    
    # 5 object detection
    if "5" in inp:
        with mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.99, model_name='Shoe') as objectron:
            objectron_results = objectron.process(image)
    
    # 6 Pose Detection
    if "6" in inp:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_results = pose.process(image)
    
    # 7 Selfie Segmentation
    if "7" in inp:
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            selfie_results = selfie_segmentation.process(image)
    
    if "8" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            vol_con_results = hands.process(image)
    
    if "9" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            fing_count_results = hands.process(image)
        
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 1 face detection
    if "1" in inp and face_det_results.detections:
        for detection in face_det_results.detections:
            mp_drawing.draw_detection(image, detection)
    
    # 2 face mesh detection
    if "2" in inp and face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
    
    # 3 hand detection
    if "3" in inp and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    # 4 holistic detectiom
    if "4" in inp:
        mp_drawing.draw_landmarks(image, holistic_results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    
    # 5 object detection
    if "5" in inp and objectron_results.detected_objects:
        for detected_object in objectron_results.detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
    
    # 6 pose detection
    if "6" in inp:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 7 selfie segmentation
    if "7" in inp:
        condition = np.stack((selfie_results.segmentation_mask,) * 3, axis=-1) > 0.1
        
        # path = ''
        # bg_image = cv2.imread(path)
        # bg_image = cv2.GaussianBlur(image, (55, 55), 0)
        
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        image = np.where(condition, image, bg_image)
    
    # 8 volume change
    if "8" in inp and vol_con_results.multi_hand_landmarks:
        hand = vol_con_results.multi_hand_landmarks[0]
        landMarkList = []
        for id, landMark in enumerate(hand.landmark):
            imgH, imgW, imgC = image.shape  # height, width, channel for image
            xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
            landMarkList.append([id, xPos, yPos])
        
        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
        
        if(len(landMarkList) != 0):
            x1, y1 = landMarkList[4][1], landMarkList[4][2]
            x2, y2 = landMarkList[8][1], landMarkList[8][2]
            length = math.hypot(x2-x1, y2-y1)
            #print(length)
    
            volumeValue = np.interp(length, [50, 250], [-65.25, 0.0]) #coverting length to proportionate to volume range
            volume.SetMasterVolumeLevel(volumeValue, None)
            
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
    if "9" in inp:
        if fing_count_results.multi_handedness:
            label = fing_count_results.multi_handedness[0].classification[0].label  # label gives if hand is left or right
        
        if fing_count_results.multi_hand_landmarks:
            hand = fing_count_results.multi_hand_landmarks[0]
            landMarkList = []
            for id, landMark in enumerate(hand.landmark):
                imgH, imgW, imgC = image.shape  # height, width, channel for image
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos, label])
                
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
            count = 0
            
            if(len(landMarkList) != 0):
                if landMarkList[4][3] == "Right" and landMarkList[4][1] > landMarkList[3][1]:       #Right Thumb
                    count = count+1
                elif landMarkList[4][3] == "Left" and landMarkList[4][1] < landMarkList[3][1]:       #Left Thumb
                    count = count+1
                if landMarkList[8][2] < landMarkList[6][2]:       #Index finger
                    count = count+1
                if landMarkList[12][2] < landMarkList[10][2]:     #Middle finger
                    count = count+1
                if landMarkList[16][2] < landMarkList[14][2]:     #Ring finger
                    count = count+1
                if landMarkList[20][2] < landMarkList[18][2]:     #Little finger
                    count = count+1
            
            cv2.putText(image, str(count), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
    
    cv2.imshow('Window', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()