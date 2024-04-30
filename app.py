import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load your own pre-trained model from the h5 file
model = tf.keras.models.load_model('new model.h5')

# Define function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

# Function to make predictions
def predict(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    return prediction

# Streamlit app
def main():
    st.title("Retinal Disease Classification WebApp")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        with st.spinner('Predicting...'):
            prediction = predict(uploaded_file)
            predicted_index = np.argmax(prediction)
            st.markdown("---")

        # Get the class labels (you need to adjust this based on your model's output)
        # For example, if your model returns probabilities for multiple classes, you might have a list of class names.
        class_names = {0:'Diabetic Retinopathy', 1:'Myopia', 2:'Normal', 3:'Tesselation'}  # Replace with your actual class names

        # Display predictions
        st.subheader('Predictions:')
        diagnosis = f"{class_names[predicted_index]}"
        if diagnosis != 'Normal':
            st.warning(diagnosis)
        else:
            st.success(diagnosis)
            st.balloons()

if __name__ == "__main__":
    main()
