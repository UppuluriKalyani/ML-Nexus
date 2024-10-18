from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

train_dir = 'dataset/train'

model = load_model('vehicle_model.h5')

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1/255)

classes =['bus', 'family sedan', 'fire engine', 'heavy truck', 'jeep', 'minibus', 'racing car', 'SUV', 'taxi', 'truck']

train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(600, 600),  
        batch_size=batch_size,
        classes = classes,
        class_mode='categorical'
)

loaded_model = tf.keras.models.load_model('vehicle_model.h5')

def predict_vehicle_with_loaded_model(img_path):
    img = image.load_img(img_path, target_size=(600, 600))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = loaded_model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_label = list(train_generator.class_indices.keys())[class_idx]
    return class_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print("Received POST request")
        if 'file' not in request.files:
            print("No file part in request")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            print(f"Saving file to {file_path}")
            file.save(file_path)
            label = predict_vehicle_with_loaded_model(file_path)
            return render_template('app.html', label=label, file_path=file.filename)
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)