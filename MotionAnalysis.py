import os
import cv2
import numpy as np
from keras.models import model_from_json, load_model
from keras.preprocessing import image

## My model
model = load_model("model.h5")
model.load_weights('weights.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while (cap.isOpened()):

    # Captures frame and returns boolean value and captured image
    ret,test_img=cap.read()
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:

        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        #Cropping face area from image i.e. 
        roi_gray=gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        ## Pass this pixels to trained model to get prediction
        predictions = model.predict(img_pixels)

        #Get the label of prediction
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Detection ',resized_img)

    #If you want to exit press 'q' character.
    if cv2.waitKey() == ord('q'):
        break
## Release resources
cap.release()
cv2.destroyAllWindows