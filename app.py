
from streamlit_webrtc import webrtc_streamer,RTCConfiguration
import streamlit_webrtc
import av
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import onnxruntime
import numpy as np
import cv2
from PIL import Image
import os
from keras.models import model_from_json
import keras.backend as K
from keras.src.saving import serialization_lib
from io import BytesIO
serialization_lib.enable_unsafe_deserialization()

onnx_model = onnxruntime.InferenceSession(r'C:\Users\07032\Downloads\maths_eq_seg.onnx')

with open(r"C:\Users\07032\python\projects\captch generator\models\eq_extractor_model_2.json","r") as f:
    model_json = f.read()
    eq_extractor = model_from_json(model_json,custom_objects={"safe_mode": False})
    eq_extractor.load_weights(r"C:\Users\07032\Downloads\eq_extractor_best3.h5")

char_list = ["0","1","2","3","4","5","6","7","8","9","+","-","x","(",")"]
 
def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst
    
def answer(listt):
    new_li=[]
    prev=0
    flag=0
    for i in listt:
        if i.isnumeric():
            flag=1
            prev=prev*10+int(i)
        else:
            if flag:
                new_li.append(str(prev))
            new_li.append(i)
            prev=0
            flag=0
    if flag:
        new_li.append(str(prev))
    if len(new_li)==0:
        return ""
    listt=new_li
    try:
        if "(" in listt:
            for i in range(len(listt)):
                if listt[i]==")":
                    maxi=i
                    break
            for i in range(maxi,-1,-1):
                if listt[i]=="(":
                    mini=i
                    break
            return answer(listt[:mini]+[answer(listt[mini+1:maxi])]+listt[maxi+1:])
        else:
            new_list=[]
            for i in range(len(listt)):
                if listt[i]=="x":
                    new_list[-1]=int(new_list[-1])*int(listt[i+1])
                elif listt[i-1]!="x" and listt[i]!="-" and listt[i]!="+":
                    new_list.append(int(listt[i]))
                elif listt[i]=="+" or listt[i]=="-":
                    new_list.append(listt[i])
            for i in range(len(new_list)):
                if new_list[i]=="+":
                    new_list[i+1]+=new_list[i-1]
                elif new_list[i]=="-":
                    new_list[i+1]=new_list[i-1]-new_list[i+1]
            return str(new_list[-1])
    except:
        return "wrong eq"

def print_text_on_image(np_frm,for_display, bboxes):
    image=np.zeros((480,900,3),dtype="uint8")
    for xc,yc,w,h in bboxes:
        x=max(int(xc-w//2),0)
        y=max(int(yc-h//2),0)
        w=int(w)
        h=int(h)
        img=np_frm[y:y+h,x:x+w]
        w1,h1= img.shape
        if w1>20 and h1>20:
            strr=prediction(img)  
            if strr!="":
                np_frm=cv2.rectangle(np_frm, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            font_scale = 1
            font_thickness = 2
            ans=answer(list(strr))
            cv2.putText(image, strr+" = "+str(ans), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
    return np_frm,image

def prediction(img):
    img=resize_image(img,(384,96))
    img=img/255.
    img=np.abs(img)

    img=np.expand_dims(img,axis=-1)
    img=np.expand_dims(img,axis=0)
    predict = eq_extractor.predict(img,verbose=0)
    out = K.get_value(K.ctc_decode(predict, input_length=np.ones(predict.shape[0])*predict.shape[1],
                             greedy=True)[0][0])
    strr=""
    for x in out:
        strr=""
        for p in x:
            if int(p) != -1:
                strr+=char_list[int(p)]
    return strr

def transform(image):
    image_640_640=resize_image(image.copy(),(640,640))
    image_yolo_pred=yolo_onxx_preprocessor(image_640_640)
    output = onnx_model.run(None, {'images': image_yolo_pred})
    bounding_boxes, confidences, class_probs = decode_yolo_output(output)
    result_image,printed=print_text_on_image(image_640_640,image.copy(),bounding_boxes)
    return result_image,printed
def non_max_suppression(boxes, scores, threshold=0.1):
    if len(boxes) == 0:
        return []

    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2

    areas = (x_max - x_min) * (y_max - y_min)

    order = scores.argsort()[::-1]

    selected_indices = []

    while len(order) > 0:
        i = order[0]
        selected_indices.append(i)

        xx1 = np.maximum(x_min[i], x_min[order[1:]])
        yy1 = np.maximum(y_min[i], y_min[order[1:]])
        xx2 = np.minimum(x_max[i], x_max[order[1:]])
        yy2 = np.minimum(y_max[i], y_max[order[1:]])

        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        suppressed_indices = np.where(iou <= threshold)[0]
        order = order[suppressed_indices + 1]

    return selected_indices

def bb_transform2(x):
    return x+20


def decode_yolo_output(output, num_classes=1):
    confidence_threshold=0.1
    iou_threshold=0.1
    output=output[0]
    bounding_boxes = output[:, :, :4]
    bounding_boxes[:, :, 2] = bb_transform2(bounding_boxes[:, :, 2])
    bounding_boxes[:, :, 3] = bb_transform2(bounding_boxes[:, :, 3])
    
    confidences = output[:, :, 4]
    class_probs = output[:, :, 5:]

    mask = confidences > confidence_threshold
    bounding_boxes = bounding_boxes[mask]
    confidences = confidences[mask]
    class_probs = class_probs[mask]

    selected_indices = non_max_suppression(bounding_boxes, confidences, iou_threshold)

    bounding_boxes = bounding_boxes[selected_indices]
    confidences = confidences[selected_indices]
    class_probs = class_probs[selected_indices]

   

    return bounding_boxes, confidences, class_probs


    
def yolo_onxx_preprocessor(image):
    image=image.astype("uint8")
    image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)  
    image = image.astype(np.float32) / 255.0  
    image = np.transpose(image, (2, 0, 1))  
    image = np.expand_dims(image, axis=0)  
    return image

def resize_image(image, imgSize):
    img=image.copy()
    if len(img.shape)==3 and img.shape[2]==3: 
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  
    
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) # INTER_CUBIC interpolation best approximate the pixels image
    # #                                                            # see this https://stackoverflow.com/a/57503843/7338066
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img

    return target


def draw_boxes(img, boxes):
    image=img.copy()
    for box in boxes:
        x, y, w, h = box
        x=x-w//2
        y=y-h//2
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    return image

def find_dominant_color(image):
        
        width, height = 150,150
        image = image.resize((width, height),resample = 0)
        pixels = image.getcolors(width * height)
        sorted_pixels = sorted(pixels, key=lambda t: t[0])
        dominant_color = sorted_pixels[-1][1]
        return dominant_color


def main():

    start = st.button('Start')
    stop = st.button('Stop')

    if 'camera' not in st.session_state:
        st.session_state.camera = False

    if start:
        st.session_state.camera = True
    if stop:
        st.session_state.camera = False



    image=st.camera_input("image")

    # cap = cv2.VideoCapture(1)

    # while st.session_state.camera:
    #     ret, frame = cap.read()
    #     if not ret:
    #         st.write("Failed to grab frame")
    #         break
    if image is not None:
        image= Image.open(image)

        # Convert to NumPy array
        image = np.array(image)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
        result_image,printed=transform(frame)
        # result_image/=255.
        # printed/=255.
        cols=st.columns(3)
        
        with cols[0]:
            original = st.empty()
            frame=cv2.resize(frame,(320,240))
            original.image(frame,"original")
        with cols[1]:
            line_img = st.empty()
            result_image=result_image.astype("uint8")
            result_image=cv2.resize(result_image,(320,240))
            line_img.image(result_image,"original")
        with cols[2]:
            solved = st.empty()
            printed=cv2.resize(printed,(320,240))
            solved.image(printed)

    # else:
        # cap.release()

if __name__ == '__main__':
    main()
