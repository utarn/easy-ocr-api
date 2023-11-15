from flask import Flask, request, jsonify, send_file
import easyocr
import base64
import numpy as np
import cv2
from PIL import Image
import dlib
import io

# Load Dlib's pre-trained facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def detect_eyes(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Initialize min and max coordinates for eyes
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        # Loop through each eye landmark (36 to 47)
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        # Draw a black rectangle that covers both eyes
        cv2.rectangle(img_np, (min_x, min_y), (max_x, max_y), (0, 0, 0), -1)

    return img_np

app = Flask(__name__)
reader = easyocr.Reader(['en','th'], download_enabled=False)  # Assuming English text

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        # Decode the image from base64
        data = request.json
        image_b64 = data['image']
        decoded_image = base64.b64decode(image_b64)

        # Convert the decoded bytes to a numpy array
        nparr = np.frombuffer(decoded_image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # Check if the image has an alpha channel (RGBA) and convert to RGB
        if img_np.shape[-1] == 4:  # Image has 4 channels
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.convert('RGB')
            img_np = np.array(img_pil)

        # OCR processing
        results = reader.readtext(img_np)

        # Draw rectangles on even segments of the detected text
        overlay = img_np.copy()
        for (bbox, text, prob) in results:
            top_left, top_right, bottom_right, bottom_left = bbox
            segment_length = (top_right[0] - top_left[0]) / 10  # Length of each segment

            for i in range(10):
                if i % 2 == 1:  # Draw only on even segments (0-indexed)
                    start_x = int(top_left[0] + i * segment_length)
                    end_x = int(start_x + segment_length)

                    start_y = int(top_left[1])
                    end_y = int(bottom_left[1])

                    cv2.rectangle(img_np, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)

        # Blend the overlay (50% opacity)
        # cv2.addWeighted(overlay, 0.5, img_np, 0.5, 0, img_np)

        img_np = detect_eyes(img_np)
        # Convert the modified image back to base64
        _, buffer = cv2.imencode('.jpg', img_np)

        #img_base64 = base64.b64encode(buffer).decode()

        # Convert results to a simple list
        #text_list = [text for _, text, _ in results]

        # Return the results along with the modified image
        # return jsonify({"ocr_result": text_list, "image": img_base64})
        #return jsonify({"image": img_base64})
        img_bytes = buffer.tobytes()

        # Set the content type and headers for the response
        response_headers = {
            'Content-Type': 'image/jpeg',
            'Content-Disposition': 'attachment; filename=image.jpg',
        }

        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='image.jpg'
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
