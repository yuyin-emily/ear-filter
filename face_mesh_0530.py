from math import sqrt
import cv2
import math
import numpy as np
import mediapipe as mp
import requests
import base64
import time
import requests

# 設定GitHub倉庫的相關資訊
username = 'emilychen0716'
repo_name = 'face_filter/main'


# 設定 GitHub 帳戶驗證，使用個人訪問令牌（Personal Access Token）
access_token = ''
headers = {'Authorization': f'token {access_token}'}
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
eartype="bear"
imageplace="test"

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.7) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_FACEMESH_CONTOURS, # old as CONNECTIONS
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as face_mesh:
  while cap.isOpened():
    success, image = cap.read(cv2.IMREAD_UNCHANGED)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    

    if results.multi_face_landmarks:
      if len(results.multi_face_landmarks)>1:
        print(len(results.multi_face_landmarks))
      for face_landmarks in results.multi_face_landmarks:
        p1=[int(face_landmarks.landmark[67].x*image_width),int(face_landmarks.landmark[67].y*image_height)]
        p2=[int(face_landmarks.landmark[297].x*image_width),int(face_landmarks.landmark[297].y*image_height)]
        dis=3*int(sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))
        delta_x=(p1[0]-p2[0])*3
        delta_y=(p1[1]-p2[1])*3
        
      #print(eartype)
      key = cv2.waitKey(1)
      

      if key == 13:
        print("change")
        print(eartype)
        if eartype == "bear":
            eartype = "cat"
        elif eartype == "cat":
            eartype = "bear"
        print(eartype)
      
      if eartype == "cat":
        ear = cv2.imread('catear.png', cv2.IMREAD_UNCHANGED)
      elif eartype == "bear":
        ear = cv2.imread('bearear.png', cv2.IMREAD_UNCHANGED)

      ear = cv2.resize(ear, (dis,int(dis/ear.shape[0]*ear.shape[1])))
      angle = math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0]))
      ear = rotate_image(ear,-angle)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
      x=p1[1]-dis//3*2
      y=p1[0]-dis//3
      if x+ear.shape[0]<image.shape[0] and y+ear.shape[0]<image.shape[1] and x>0 and y>0:
        bg=np.zeros(image.shape,np.uint8)
        bg[x:x+ear.shape[0], y:y+ear.shape[1]] = ear
        image = cv2.addWeighted(image,1,bg,1,-5)

      if key == 32:
        print("Download!")
        current_time = time.strftime("%H%M")
        cv2.imwrite(str(current_time)+'.png', image)
        print("圖片下載完成！")
        file_path = str(current_time)+'.png'  # 要上傳的圖片檔案路徑

        # 讀取圖片檔案的二進位數據
        with open(file_path, 'rb') as file:
            file_data = file.read()

        # 設定API端點和標頭
        url = f'https://api.github.com/repos/{username}/{repo_name}/contents/{file_path}'

        # 建立請求的有效負載
        payload = {
            'message': 'Upload image',
            'content': file_data
        }

        # 發送POST請求上傳圖片
        response = requests.post(url, headers=headers, json=payload)

        # 檢查回應狀態碼
        if response.status_code == 201:
            print('Image uploaded successfully.')
        else:
            print('Failed to upload image.')
            print('Response:', response.json())
        
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()