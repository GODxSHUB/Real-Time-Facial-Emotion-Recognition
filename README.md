# Real-Time Facial Emotion Recognition 🎭

A machine learning and computer vision project that detects human faces in real-time via a webcam feed and classifies their current emotional state using a deep learning model.

## 📌 Project Overview
This project leverages **Python, OpenCV, and TensorFlow/Keras** to predict facial expressions on the fly. It captures live video frames, detects facial coordinates, and feeds the cropped face data into a pre-trained neural network (`.h5` model) to output the predicted emotion.

### Recognized Emotions:
* Angry 😠
* Disgust 🤢
* Fear 😨
* Happy 😄
* Neutral 😐
* Sad 😢
* Surprise 😲

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning Framework:** TensorFlow / Keras
* **Environment:** Jupyter Notebook (for training), Python Scripts (for real-time inference)

## 🚀 Features
* **Real-time Detection:** Hooks directly into your local webcam for live inference.
* **Lightweight & Fast:** Optimized for quick frame-by-frame processing without heavy lag.
* **Custom Trained Model:** Built and trained using a Convolutional Neural Network (CNN) architecture.

data :- you will get the data on kaggle
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

2. Install dependencies
Make sure you have Python installed. Install the required libraries using the provided requirements.txt file:

Bash
pip install -r requirements.txt

3. Run the real-time detector
Ensure your webcam is connected and accessible, then execute the script:

Bash
python realtimedetection.py
(Press q to quit the webcam feed at any time.)

🧠 Model Training (Optional)
If you want to tweak the architecture or retrain the model from scratch on your own dataset:

Open trainmodel.ipynb in Jupyter Notebook or Google Colab.

Run the cells to preprocess the data, train the CNN, and output a new .h5 model weight file.

👨‍💻 Author
Shubham Kumar (Computer Science Engineering, C.V. Raman Global University (Class of 2028))

GitHub: @GODxSHUB

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

Give a STAR if u like it. 
fork the repo if you want

***





