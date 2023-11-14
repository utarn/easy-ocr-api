from flask import Flask, request, jsonify
import easyocr
import base64
from PIL import Image
import io

app = Flask(__name__)
reader = easyocr.Reader(['en'])  # Assuming English text

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        # Decode the image from base64
        data = request.json
        image_b64 = data['image']
        decoded_image = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(decoded_image))

        # OCR processing
        results = reader.readtext(image)

        # Convert results to a simple list
        text_list = [text for _, text, _ in results]

        # (Optional) Convert the image back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Return the results
        return jsonify({"ocr_result": text_list, "image": img_str})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

