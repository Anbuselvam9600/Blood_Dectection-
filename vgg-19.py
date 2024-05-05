from flask import Flask, render_template, request
from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.preprocessing import image  # Add this import statement
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input


app = Flask(__name__)

# Define the path to the directory containing blood group images
file_path = '/home/karuppasamy/Downloads/archive (1)/dataset_blood_group'

# Check if the directory exists
if not os.path.isdir(file_path):
    print("Error: Directory '{}' does not exist.".format(file_path))
    exit()

# Get the list of blood group names from the directory
name_class = sorted(os.listdir(file_path))
print("List of blood group names:", name_class)

# Assigning blood group names to indices
blood_group_names = {index: blood_group for index, blood_group in enumerate(name_class)}
print("Blood group indices with names:", blood_group_names)

# Load the trained model
model = None

def load_model():
    global model
    try:
        # Load VGG19 model
        vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

        # Remove the last layers of VGG-19
        output = vgg19_base.layers[-1].output

        # Add Global Average Pooling layer
        output = GlobalAveragePooling2D()(output)

        # Create a new model with VGG-19 base and global average pooling
        pretrained_model = Model(inputs=vgg19_base.input, outputs=output)

        # Freeze the layers
        for layer in pretrained_model.layers:
            layer.trainable = False

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

        # Save the model weights
        model.save_weights("model_blood_group_detection.h5")

        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)

# Function to analyze the image and predict the blood group with confidence scores
def analyze_image(file_path):
    try:
        img = image.load_img(file_path, target_size=(256, 256))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        result = model.predict(x)[0]  # Get the prediction result for the first image (batch size = 1)
        print(result)
        value=(result*100).astype('int')
        print(value)
        confidence_scores = {blood_group_names[i]: score for i, score in enumerate(result)}
        predicted_blood_group = max(confidence_scores, key=confidence_scores.get)
        print(predicted_blood_group)
        return predicted_blood_group
    except Exception as e:
        print("Error analyzing image:", e)
        return 'Unknown', {}


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
            # Save the uploaded file to the 'uploads' directory
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Analyze the uploaded image
            predicted_blood_group = analyze_image(file_path)
            return render_template('result.html', blood_group=predicted_blood_group)
    return render_template('index.html')

if __name__ == '__main__':
    load_model()  # Load the model when the application starts
    app.run(debug=True)
