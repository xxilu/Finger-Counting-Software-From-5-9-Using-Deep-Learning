import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import keras

# Load your model, import libraries, etc.

file_model = 'resnet50_transfer_box.h5'
loaded_model = keras.models.load_model(file_model)


def test_predict(test_img_path):  # Change the argument name
    labels = ["5", "6", "7", "8", "9"]
    img = cv2.imread(test_img_path)  # Use the file path directly
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (128, 128))
    img_arr = np.asarray(img_resize) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred, bbox = loaded_model.predict(img_arr)  # Use loaded_model directly
    itemindex = np.argmax(pred)
    prediction = itemindex

    # Vẽ bounding box
    x_min, y_min, x_max, y_max = bbox[0]
    h, w, _ = img_rgb.shape
    x1, y1 = int(x_min * w), int(y_min * h)
    x2, y2 = int(x_max * w), int(y_max * h)
    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return labels[prediction], np.max(pred), img_rgb


def main():
    st.title("Image Prediction with Bounding Box")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_img_path = "temp_image.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        image = plt.imread(temp_img_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions when a button is clicked
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                predicted_class, probability, annotated_image = test_predict(temp_img_path)
                st.success('Prediction completed!')

                # Display prediction information
                st.write("Predicted class:", predicted_class)
                st.write("Probability:", probability * 100, "%")

                # Display annotated image
                st.image(annotated_image, caption="Annotated Image", use_column_width=True)

        # Delete the temporary image file
        os.remove(temp_img_path)


if __name__ == "__main__":
    main()
