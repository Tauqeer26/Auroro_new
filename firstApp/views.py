from django.shortcuts import render
from django.urls import include, re_path
from django.urls import re_path as url
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
from pytest import Session
import tensorflow as tf
import json
from tensorflow import Graph
import cv2

img_height, img_width=256,192
with open('./models/label.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/aur_1.h5')#Final_2.h5


def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    #print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    print(filePathName)
    testimage='.'+filePathName
    #img = image.load_img(testimage, target_size=(img_height, img_width))
    try:
        img = cv2.imread(testimage)
        
        if img=='None':
            print('No picture')
        img = cv2.resize(img,(img_height, img_width))
    except:
        print("No picture")
    
    
    #x = image.img_to_array(img)
    #x=x/255
    #x=cv2.resize(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            import numpy as np
            #img = cv2.resize(img,(img_height, img_width))
            predi=model.predict(np.array([img]).astype(np.float32))
            print(predi)

    import numpy as np
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    print(predictedLabel)
    context={'filepath':filePathName, 'predictedLabel':predictedLabel}
    print(context)
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 