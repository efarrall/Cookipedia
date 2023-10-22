#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import random
import os
import shutil


# In[2]:


#Packages for segmenting
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import cv2
import numpy as np
import torch 
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("..")


# In[172]:


frameWidth = 960
frameHeight = 540
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

success, img = cap.read()
cv2.imshow("Result", img)


# In[145]:


#Initialize image and settings idk
image_path = r"C:\Users\farra\Downloads\pumkinbroc.jpg"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image)

img_org = cv2.imread(image_path)
img_org = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



#original image 
#plt.figure(figsize=(10,10))
#plt.imshow(image)
#plt.axis('on')
#plt.show()

sam = sam_model_registry["vit_b"](checkpoint=r"C:\Users\farra\Downloads\sam_vit_b_01ec64.pth")
sam.to(device=device)
pass


# In[141]:


# generate


mask_generator = SamAutomaticMaskGenerator(sam)


masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())




# In[142]:


# Create a directory to save the segmented images
output_dir = r'C:\Users\farra\Downloads\vegoutputs2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[156]:


# Save the segmented parts of the image as individual images
segmented_list = []
for i, ann in enumerate(masks):
    mask = ann['segmentation']
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    roi = image[y:y+h, x:x+w]
    
    original_roi = img_org[y:y+h, x:x+w]
    if abs(h) > image.shape[0]/10 and abs(w) > image.shape[1]/10:
        cv2.imwrite(os.path.join(output_dir, f'segmented_{i}.jpg'), original_roi)
        img = Image.open((os.path.join(output_dir, f'segmented_{i}.jpg')))
        segmented_list += [img]

print(len(segmented_list))
print(segmented_array)
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


# In[ ]:


loaded_model = tf.keras.saving.load_model(r'C:\Users\farra\OneDrive\Desktop\Machine_Learning\InceptionV3_Vegetabletoptune_1epochs')


# In[150]:


#Image preproccessing
from PIL import Image

def preprocess_imageload(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size, Image.ANTIALIAS)

    # If the image is smaller than the target size, add padding
    if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
        new_img = Image.new("RGB", target_size)
        new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))
        img = new_img

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize pixel values if necessary
    #img_array = img_array / 255.0  # Normalize pixel values between 0 and 1

    return img_array

def preprocess_imagegood(image, target_size=(224, 224)):
    image_array = np.array(image, dtype=np.uint8)

    # Create a Pillow image from the NumPy array
    pillow_image = Image.fromarray(image_array)
    img = pillow_image.resize(target_size, Image.ANTIALIAS)

    # If the image is smaller than the target size, add padding
    if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
        new_img = Image.new("RGB", target_size)
        new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))
        img = new_img

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize pixel values if necessary
    #img_array = img_array / 255.0  # Normalize pixel values between 0 and 1

    return img_array



# In[ ]:


# Example usage:
image_path = r"C:\Users\farra\Downloads\cabbage.jpg"
show_img = preprocess_image(image_path)
plt.imshow(show_img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
preprocessed_image = show_img.reshape(1, 224, 224, 3)
print(preprocessed_image.shape)


# In[157]:


#Segmented usage
processed_list = []
for img in segmented_list:
    processed_list += [preprocess_imagegood(img)]
print(len(processed_list))
for i in range(0, len(segmented_list)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(segmented_list[i])
    plt.axis('off')


# In[31]:


class_list = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


# In[163]:


plt.figure(figsize=(10,10))
outputlst = []
for i in range(0, len(processed_list)):   
    img = processed_list[i]
    img = img.reshape(1, 224, 224, 3)
    predictions = loaded_model(img, training=False)
    softout = tf.nn.softmax(predictions)
    decoded_predictions = class_list[np.argmax(softout)]
    outputlst += [decoded_predictions]
    print(decoded_predictions)
 

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(segmented_list[i])
    plt.axis('off')




#this for if using original inceptionV3: decoded_predictions = decode_predictions(predictions, top=1)
#ax = plt.figure(figsize = (10,10))
#for i in range(20):
#    plt.subplot(5, 4, i+1)
#    plt.imshow(image[i]/255.0)
#    plt.title(predictions[i])
#    plt.axis('off')


# In[164]:


outputlst = set(outputlst)
print(outputlst)


# In[170]:


#LLM
import openai

api_key = "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"


ingredients = outputlst


cuisine = input("What cuisine would you like to make? (any if no preference): ")
if cuisine == "any":
    cuisine = ""


input_prompt = f"I have {', '.join(ingredients)} and want to make a {cuisine} dish. Please provide a recipe for it."

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=input_prompt,
    max_tokens=200,
    api_key=api_key
)
recipe = response.choices[0].text
print("Generated Recipe:")
print(recipe)


# In[173]:


from flask import Flask, request, jsonify, render_template
app = Flask(__name__)



# Import and load your machine learning model here
# Example: 
# from your_module import your_model
# your_model.load_weights('model_weights.h5')

your_model = tf.keras.saving.load_model(r'C:\Users\farra\OneDrive\Desktop\Machine_Learning\InceptionV3_Vegetabletoptune_1epochs')
#sam.load_weights('/Users/marmik/Downloads/sam_vit_b_01ec64.pth')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        try:
    

    # Get the input data as JSON from the POST request
            data = request.get_json()

        # Validate and preprocess the input data if needed
            input_data = preprocess_data(data)


        # Format the prediction result as needed
            result = format_prediction(outputlst)
            return jsonify({'result': result})

        except Exception as e:
            return jsonify({'error': str(e)})
            return {'result': 'Prediction result'}
        



# Front-end API endpoint
@app.route('/frontend_api', methods=['POST'])
def frontend_api():
    try:
        data = request.get_json()
        # Process data, interact with your model, etc.
        # You can add your specific logic here based on the front-end requirements
        result = {'message': 'Front-end API call successful'}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})
                       

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




