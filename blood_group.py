from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Model

app = Flask(__name__)

# Mapping dictionary for blood group names
file_path = '/home/karuppasamy/Downloads/archive (1)/dataset_blood_group'
name_class = os.listdir(file_path)
print("List of blood group names:", name_class)  # Print the list of blood group names

# Assigning blood group names to indices
blood_group_names = {index: blood_group for index, blood_group in enumerate(name_class)}
print("Blood group indices with names:", blood_group_names)

# Load the trained model
model = None  # Define it globally to use it in multiple routes

def load_model():
    global model
    # Load your trained model here
    try:
        pretrained_model = ResNet50(
            input_shape=(256, 256, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        pretrained_model.trainable = False

        inputs = pretrained_model.input
        x = Dense(128, activation='relu')(pretrained_model.output)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(len(name_class), activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.load_weights("/home/karuppasamy/model_blood_group_detection.h5")  # Load model weights
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)

# Function to process the uploaded image
def process_image(file_path):
    try:
        img = image.load_img(file_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except Exception as e:
        print("Error processing image:", e)
        return None

# Function to analyze the image and predict the blood group
def analyze_image(file_path):
    try:
        processed_image = process_image(file_path)
        if processed_image is not None:
            result = model.predict(processed_image)
            predicted_blood_group_index = np.argmax(result)
            predicted_blood_group = blood_group_names.get(predicted_blood_group_index, 'Unknown')
            return predicted_blood_group
        else:
            return 'Unknown'
    except Exception as e:
        print("Error analyzing image:", e)
        return 'Unknown'

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            predicted_blood_group = analyze_image(file_path)
            return render_template('result.html', blood_group=predicted_blood_group)
    return render_template('index.html')

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(debug=True)
