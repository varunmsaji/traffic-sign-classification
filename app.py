import streamlit as st 
from tensorflow import keras
from keras.preprocessing import image
import numpy as np

model = keras.models.load_model('traffic.h5')

st.set_page_config(page_title="traffic sign classification",layout="centered")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
button = st.button('predict')
all_lables = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)',
              'Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)',
              'Speed limit (120km/h)','No passing','No passing for vechiles over 3.5 metric tons',
              'Right-of-way at the next intersection','Priority road','Yield','Stop','No vechiles',
              'Vechiles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left',
              'Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right',
              'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',
              'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',
              'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory',
              'End of no passing','End of no passing by vechiles over 3.5 metric']

if uploaded_file and button:
    img = image.load_img(uploaded_file,target_size=(50,50))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array /255.0
    result = model.predict(img_array)
    final_result = np.argmax(result)
    st.write(all_lables[final_result])