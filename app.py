from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load a pretrained model (example: MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the image file with Pillow and ensure it is in RGB mode
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # Resize to the size expected by the model
        image_array = np.array(image)  # Convert the image to a numpy array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)  # Preprocess the image
        
        # Make predictions
        predictions = model.predict(image_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
        
        # Prepare the result
        result = [{'class': pred[1], 'probability': float(pred[2])} for pred in decoded_predictions[0]]
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
