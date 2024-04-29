from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model = load_model("trained_model1.h5")

@app.route("/", methods=['POST'])
def predictFromApi():
    file = request.files['image']
    print(file)
    file.save(secure_filename("New.jpg"))

    img = image.load_img("New.jpg", target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.reshape(-1, 224, 224, 3)
    img /= 255.0

    prediction = model.predict(img)

    class_labels = ['Acne', 'HairLoss', 'Skin Cancer', 'unidentified']  # List of class labels
    predicted_label_index = np.argmax(prediction)
    if np.max(prediction) > 0.5:  # Assuming predictions for a single sample
        predicted_label = class_labels[predicted_label_index]
    else:
        predicted_label = class_labels[3]
    response = jsonify({"predicted": predicted_label})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    app.run(debug=True, port=2000)
