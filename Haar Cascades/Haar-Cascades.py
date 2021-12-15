import cv2

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('./haarcascades/mouth.xml')
crop_faces = 0
crop_eyes = 0
crop_mouth = 0

vid = cv2.VideoCapture(0)
while(True):
    ret,frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, 'faces', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)
        crop_faces=frame[y:y+h,x:x+w]
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in eyes:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, 'eyes', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        crop_eyes=frame[y:y+h,x:x+w]
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 40)
    for (x,y,w,h) in mouth:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame, 'mouth', (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        crop_mouth=frame[y:y+h,x:x+w]
       
    cv2.imshow('Detected!', frame)
    cv2.imshow('Faces Detected!', crop_faces)
    cv2.imshow('Eyes Detected!', crop_eyes)
    cv2.imshow('Mouth Detected!', crop_mouth)
    
    
    
    keyboard = cv2.waitKey(38)
    if keyboard == 'q' or keyboard == 27:
        break
cv2.destroyAllWindows()
