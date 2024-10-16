from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import base64
from deoldify import device
from deoldify.device_id import DeviceId
import torch
import traceback
from pyngrok import ngrok

# Use GPU if available
if torch.cuda.is_available():
    device.set(device=DeviceId.GPU0)
else:
    print('GPU not available. Using CPU.')

from deoldify.visualize import get_image_colorizer

colorizer = get_image_colorizer(artistic=True)

def deoldify_process(img):
    # Ensure the image is in RGB mode
    img = img.convert('RGB')
    
    # Get the path to save the temporary image
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        img_path = temp_file.name
        img.save(img_path)
    
    try:
        # Colorize the image
        result = colorizer.get_transformed_image(img_path, render_factor=35)
        
        # Convert the result to a PIL Image if it's not already
        if not isinstance(result, Image.Image):
            result = Image.fromarray(result)
        
        return result
    finally:
        # Clean up the temporary file
        os.unlink(img_path)

app = Flask(__name__)
CORS(app)

@app.route('/colorize', methods=['POST'])
def colorize_image():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode the base64 image data
        image_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(image_data))
        
        # Process the image with DeOldify
        colorized_img = deoldify_process(img)
        
        # Convert the colorized image to base64
        buffered = BytesIO()
        colorized_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'colorizedImageUrl': f'data:image/png;base64,{img_str}'})
    except Exception as e:
        error_message = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    # Open a ngrok tunnel to the HTTP server
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")

    # Start the Flask server
    app.run()