import cv2
import mediapipe as mp
import time

mediaHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils


video = cv2.VideoCapture(0)

with mediaHand.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hand:
    while video.isOpened():
        suc, img = video.read()
        start = time.time()
        img = cv2.cvtColor(cv2.flip(img,1),cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        res = hand.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        if res.multi_hand_landmarks:
            for landmarks in res.multi_hand_landmarks:
                print(landmarks)
                mpDraw.draw_landmarks(img,landmarks,mediaHand.HAND_CONNECTIONS)

        end = time.time()
        total = end-start

        fps = 1/total

        cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0))
        cv2.imshow('Media Hands',img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

video.release()