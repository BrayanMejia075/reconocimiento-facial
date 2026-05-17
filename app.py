from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import cv2
import numpy as np

app = Flask(__name__)

AUTHORIZED_FACE = "David.jpeg"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():

    data = request.json['image']

    image_data = data.split(',')[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:

        result = DeepFace.verify(
            frame,
            AUTHORIZED_FACE,
            enforce_detection=False,
            model_name="Facenet"
        )

        if result["verified"]:

            return jsonify({
                "success":True
            })

    except Exception as e:
        print(e)

    return jsonify({
        "success":False
    })

if __name__ == '__main__':
    app.run(debug=True)