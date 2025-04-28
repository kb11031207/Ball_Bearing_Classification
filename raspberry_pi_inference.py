import time
from typing import Counter
import numpy as np
import cv2
from PIL import Image

# Try to import tflite_runtime, fall back to tensorflow if not available
try:
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime for inference")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("Using TensorFlow Lite from main TensorFlow package")
    except ImportError:
        print("ERROR: Neither tflite_runtime nor tensorflow is installed")
        print("For Raspberry Pi, install with:")
        print("pip3 install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_armv7l.whl")
        exit(1)

# Try to import picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("ERROR: picamera2 is not installed. Install it with:")
    print("sudo apt install -y python3-picamera2")
    exit(1)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="material_model_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Model parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ["Brass", "nylon", "Steel"]

def preprocess_image(image):
    """Preprocess the image for model input"""
    # Resize image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

def classify_image(image):
    """Classify an image using the TFLite model"""
    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class_idx]
    print(f"Predicted class index: {predicted_class_idx}, confidence: {confidence}")

    return {
        'class': CLASS_NAMES[predicted_class_idx],
        'confidence': float(confidence),
        'inference_time': inference_time
    }
def run_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
    picam2.start()

    print("Press 'q' to quit")

    last_predictions = []
    VOTE_SIZE = 5

    prev_time = time.time()

    while True:
        frame = picam2.capture_array()
        display_frame = frame.copy()

        result = classify_image(frame)
        predicted_class = result['class']

        last_predictions.append(predicted_class)
        if len(last_predictions) > VOTE_SIZE:
            last_predictions.pop(0)

        most_common_prediction = Counter(last_predictions).most_common(1)[0][0]

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Overlay
        text = f"{most_common_prediction}: {result['confidence']:.2f}"
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        time_text = f"Time: {result['inference_time']*1000:.1f}ms"
        cv2.putText(display_frame, time_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display_frame, fps_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Ball Bearing Classifier', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
