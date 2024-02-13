from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import pytesseract
import torch
import numpy as np
import io
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.0.185:8081", "http://localhost:8081"],  # Zmień ten adres na adres twojej aplikacji internetowej
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
model = None

# Set the Tesseract path to the location of the Tesseract executable
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' WINDOWS
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

@app.on_event("startup")
def load_model():
    global model
    model = torch.hub.load('./yolov5', 'custom', source='local', path='last.pt', force_reload=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global model  # access the global model variable

    contents = await file.read()
    npimg = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # convert to cv2 image

    # Perform object detection
    results = model(img)

    # Extract bounding boxes
    plates = []
    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.5:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # Extract license plate
            plate = img[y1:y2, x1:x2]
            # Convert to grayscale for Tesseract
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Recognize characters
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DATAFRAME)
            filtered_data = data[(data.conf != -1) & (data.height > (y2 - y1) * 0.5)]
            text = ' '.join(filtered_data.text)

            plates.append(text)

    return JSONResponse(content={"plates": plates})

@app.get("/health")
async def healthcheck():
    return JSONResponse(content={"status": "OK", "message": "Service is running"}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import unittest
from fastapi.testclient import TestClient
from main import app

class TestDetectEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_detect_endpoint(self):
        file_data = b'\x89PNG\r\n... (dane obrazu PNG) ...'

        #wysłanie żądania POST z danym plikiem
        response = self.client.post("/detect", files={"file": ("test_image.jpg", file_data)})

        # Sprawdzenie, czy odpowiedź ma wartość OK
        self.assertEqual(response.status_code, 200)

        # Sprawdzenie, czy odpowiedź json ma poprawne dane
        data = response.json()
        self.assertIn("plates", data)
        self.assertIsInstance(data["plates"], list)

    def tearDown(self):
        #usunięcie niepotrzebnego już pliku po teście
        if os.path.exists("test_image.jpg"):
            os.remove("test_image.jpg")

if __name__ == "__main__":
    unittest.main()