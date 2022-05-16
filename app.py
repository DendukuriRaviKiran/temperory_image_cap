import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import requests
import json
from types import SimpleNamespace

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import streamlit as st
from PIL import Image





BASE_DIR = 'archive'

model = pickle.load(open('model.pkl','rb'))
features = pickle.load(open('features.pkl','rb'))
mapping = pickle.load(open('mapping.pkl','rb'))

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = 35

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    #print('---------------------Actual---------------------')
    #for caption in captions:
        #print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    return y_pred

def func(img_name):
  url = "https://api.imagga.com/v2/tags"
  querystring = {"image_url":img_name,"version":"2"}
  from array import array
  import os
  from PIL import Image
  import sys
  import time

  subscription_key = "985843655b7c4e8cadaa0d1298081bd7"
  endpoint = "https://272.cognitiveservices.azure.com/"

  computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

  headers = {
    'accept': "application/json",
    'authorization': "Basic YWNjXzMzOGRjOGM1OTUwYjQ1MTozYzA3ZjVjMGEyMWYyNmQ0MDRkZWM4NGFjODIyYWYzOQ=="
    }

  # Images used for the examples: Describe an image, Categorize an image, Tag an image, 
  # Detect faces, Detect adult or racy content, Detect the color scheme, 
  # Detect domain-specific content, Detect image types, Detect objects
  #images_folder = os.path.join (os.path.dirname(os.path.abspath(__file__)), "images")
  remote_image_url = img_name

  # Call API with remote image
  tags_result_remote = computervision_client.tag_image(remote_image_url )
  describe = computervision_client.describe_image(remote_image_url )
  # Print results with confidence score
  #print("Tags in the remote image: ")
  
  for cap in describe.captions:
          return cap.text 
def tags(img_name):
  url = "https://api.imagga.com/v2/tags"
  querystring = {"image_url":img_name,"version":"2"}

  headers = {
    'accept': "application/json",
    'authorization': "Basic YWNjXzMzOGRjOGM1OTUwYjQ1MTozYzA3ZjVjMGEyMWYyNmQ0MDRkZWM4NGFjODIyYWYzOQ=="
    }
  response = requests.request("GET", url, headers=headers, params=querystring)
  x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
  count = 0
  final_list = []
  for i in x.result.tags:
    if count < 6:
      count=count+1
      text = str(i.tag)[14:]
      text = text[:-2]
      final_list.append(text)

  return final_list

def func(img_name):
  url = "https://api.imagga.com/v2/tags"
  querystring = {"image_url":img_name,"version":"2"}
  from array import array
  import os
  from PIL import Image
  import sys
  import time

  subscription_key = "985843655b7c4e8cadaa0d1298081bd7"
  endpoint = "https://272.cognitiveservices.azure.com/"

  computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

  headers = {
    'accept': "application/json",
    'authorization': "Basic YWNjXzMzOGRjOGM1OTUwYjQ1MTozYzA3ZjVjMGEyMWYyNmQ0MDRkZWM4NGFjODIyYWYzOQ=="
    }

  # Images used for the examples: Describe an image, Categorize an image, Tag an image, 
  # Detect faces, Detect adult or racy content, Detect the color scheme, 
  # Detect domain-specific content, Detect image types, Detect objects
  #images_folder = os.path.join (os.path.dirname(os.path.abspath(__file__)), "images")
  remote_image_url = img_name

  # Call API with remote image
  tags_result_remote = computervision_client.tag_image(remote_image_url )
  describe = computervision_client.describe_image(remote_image_url )
  # Print results with confidence score
  #print("Tags in the remote image: ")
  
  for cap in describe.captions:
          return cap.text 
def tags(img_name):
  url = "https://api.imagga.com/v2/tags"
  querystring = {"image_url":img_name,"version":"2"}

  headers = {
    'accept': "application/json",
    'authorization': "Basic YWNjXzMzOGRjOGM1OTUwYjQ1MTozYzA3ZjVjMGEyMWYyNmQ0MDRkZWM4NGFjODIyYWYzOQ=="
    }
  response = requests.request("GET", url, headers=headers, params=querystring)
  x = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
  count = 0
  final_list = []
  for i in x.result.tags:
    if count < 6:
      count=count+1
      text = str(i.tag)[14:]
      text = text[:-2]
      final_list.append(text)

  return final_list


def main():
  st.title("Image-Caption-Recommender-using-NLP")
  menu = ["Image Upload","Image Url"]

  choice = st.sidebar.selectbox("Menu",menu)


  if choice == "Image Upload":
    st.subheader("Image Upload")

  elif choice == "Image Url":
    st.subheader("Image Url")
  if choice == "Image Upload":
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:
      file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
      st.write(file_details)
      st.write(generate_caption(image_file.name))
  elif choice == "Image Url":
    path = st.text_input('Image Url')
    if path:
      text = func(path)
      st.write(text)

main()