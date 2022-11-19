# -*- coding: utf-8 -*-

from __future__ import print_function


from __future__ import division

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, redirect, render_template, request
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json, load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image

global graph
graph=tf.compat.v1.get_default_graph()
#this list is used to log the predictions in the server console
predictions = np.array(["Seneca White Deer",
               "Pangolin",
               "Lady's slipper orchid",
               "Corpse Flower",
               "Spoon Billed Sandpiper",
               "Great Indian Bustard"
               ])

#this list contains the link to the predicted species
found = np.array([
    "Seneca White Deer",
               "Pangolin",
               "Lady's slipper orchid",
               "Corpse Flower",
               "Spoon Billed Sandpiper",
               "Great Indian Bustard"
        ])
app = Flask(__name__)
model = load_model("model.h5")

@app.route('/', methods=['GET'])
def index():
     # Home Page
     return render_template("index.html")
@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method== 'GET':
    return("<h6 style=\"font-face:\"Courier New\";\">No GET request herd.....</h6 >")
  if request.method== 'POST':
    # fecting the uploaded image from the post request using the id 'uploadedimg'
    f = request.files['uploadedimg']
    basepath = os.path.dirname(__file__)
    #securing the  file by creating a path in local storage
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    #Saving the uploaded image locally
    f.save(file_path)
    #loading the locally saved image
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    #converting the loaded image to image array
    x = tf.keras.utils.img_to_array(img)
    x = preprocess_input(x)
    #converting the preprocessed image to numpy array
    inp = np.array([x])
    with graph.as_default():
      #loading the saved model from training
      json_file = open('DigitalNaturalist.json')
      loaded_model_json = json_file.read()
      json_file.close()
      loaded_model = model_from_json(loaded_model_json)
      #adding weights to the trained model 
      loaded_model.load_weights("model.h5")
      #predecting the image
      preds = np.argmax(loaded_model.predict(inp),axis=1)
      
      #logs are printed to the console
      print("The predicted species is " , predictions[preds[0]])
  text = "The predicted species is " + found[preds[0]]
  return render_template("index.html", RESULT = text)


if __name__ == '__main__':
  #Threads enabled so multiple users can request simutaneously
  #debud is turned off, turn on during development to debug the errors
  #applications is binded to port 8000
  app.run(threaded = True,debug=True,port="8000")
