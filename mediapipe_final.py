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
text = '1. Face Detection\n2. Face Mesh\n3. Hand Detection\n4. Holistic Detection\n5. Pose Detection\n6. Selfie Segmentation\n7. Volume Change\n8. Fingers Count\n9. Gestures(Good Luck, Bad Luck, Ok, Peace, Yo)\n\nEnter choice: '
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
elif "10" in inp:
    mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
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
    
    '''
    # 5 object detection
    if "5" in inp:
        with mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.99, model_name='Shoe') as objectron:
            objectron_results = objectron.process(image)
    '''
    
    # 5 Pose Detection
    if "5" in inp:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_results = pose.process(image)
    
    # 6 Selfie Segmentation
    if "6" in inp:
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            selfie_results = selfie_segmentation.process(image)
    
    # 7 volume change
    if "7" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            vol_con_results = hands.process(image)
    
    # 8 Count Fingers
    if "8" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            fing_count_results = hands.process(image)
    
    # 9 Gestures
    if "9" in inp:
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            gestures = hands.process(image)
        
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 1 face detection
    if "1" in inp and face_det_results.detections:
        for detection in face_det_results.detections:
            mp_drawing.draw_detection(image, detection)
            txt = '1. Face Detection'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 2 face mesh detection
    if "2" in inp and face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
            txt = '2. Face Mesh'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 3 hand detection
    if "3" in inp and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            txt = '3. Hand Detection'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
            
    # 4 holistic detectiom
    if "4" in inp:
        mp_drawing.draw_landmarks(image, holistic_results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        txt = '4. Holistic Detection'
        cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    '''
    # 5 object detection
    if "5" in inp and objectron_results.detected_objects:
        for detected_object in objectron_results.detected_objects:
            mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)
            txt = '1. Face Detection'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    '''
    
    # 5 pose detection
    if "5" in inp:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        txt = '5. Pose Detection'
        cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 6 selfie segmentation
    if "6" in inp:
        condition = np.stack((selfie_results.segmentation_mask,) * 3, axis=-1) > 0.1

        # path = ''
        # bg_image = cv2.imread(path)
        # bg_image = cv2.GaussianBlur(image, (55, 55), 0)
        
        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        image = np.where(condition, image, bg_image)
        
        txt = '6. Selfie Segmentation'
        cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 7 volume change
    if "7" in inp and vol_con_results.multi_hand_landmarks:
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
    
            volumeValue = np.interp(length, [50, 250], [-65.25, 0.0]) #coverting length to proportionate to volume range
            volume.SetMasterVolumeLevel(volumeValue, None)
            
            cv2.circle(image, (x1, y1), 10, (10, 10, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 10, (10, 10, 0), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (10, 10, 0), 3)
            txt = '7. Volume Change'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 8 Count Fingers
    if "8" in inp and fing_count_results.multi_hand_landmarks:
        for hand_landmarks in fing_count_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fin = ''
            val = 0
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                val += 1
                fin +='Thumb '
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y:
                val += 1
                fin +='Index '
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y:
                val += 1
                fin += 'Middle '
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y:
                val += 1
                fin += 'Ring '
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y:
                val += 1
                fin += 'Pinky '
            fps = str(val) + ' Finger'
            txt = '8. Count Fingers'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
            cv2.putText(image, (fps), (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
            cv2.putText(image, (fin), (0, 75), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
    
    # 9 Gestures
    if "9" in inp and gestures.multi_hand_landmarks:
        for hand_landmarks in gestures.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fin = ''
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
                fin += 'Good Luck'
            elif hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y:
                fin += 'Bad Luck'
            elif hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
                fin += 'Ok'
            elif hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y:
                fin += 'Peace'
            elif hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                fin += 'Yo'
            cv2.putText(image, (fin), (0, 50), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)
            txt = '9. Gestures'
            cv2.putText(image, (txt), (0, 25), cv2.FONT_HERSHEY_PLAIN, 1, (10, 10, 0), 1)

    cv2.imshow('Window', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cv2.destroyAllWindows()
cap.release()