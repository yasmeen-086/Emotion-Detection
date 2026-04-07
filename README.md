# Real-Time Emotion Detection System

A real-time emotion detection project that uses your webcam to identify human emotions such as **happy, sad, angry, surprised, neutral**, etc.
Built using **computer vision** and **deep learning**, this system detects faces and predicts emotions live on video feed.

---

## 🚀 Features

* 🎥 Real-time face detection using Haar Cascade
* 🧠 Emotion recognition using Deep Learning (DeepFace)
* 📦 Works with webcam (live video feed)
* 🎯 Detects multiple emotions:

  * Happy 😊
  * Sad 😢
  * Angry 😠
  * Surprise 😲
  * Fear 😨
  * Neutral 😐
* 🧩 Bounding box with emotion label
* ⚡ Optimized for smooth real-time performance
* 🔁 Emotion stabilization to reduce flickering

---

## 🛠️ Tech Stack

* Python 🐍
* OpenCV (Computer Vision)
* DeepFace (Emotion Analysis)
* NumPy

---

## 📸 How It Works

1. Capture video stream from webcam
2. Detect faces using Haar Cascade
3. Extract face region
4. Pass face to DeepFace model
5. Predict dominant emotion
6. Display results in real-time


## 🧪 Sample Output

* Face detected with bounding box
* Emotion label displayed on screen
* Smooth real-time updates

---

## 🔥 Future Improvements

* 🎯 Face tracking for better stability
* 🌐 Web app deployment (Flask/Streamlit)
* 📊 Emotion analytics dashboard
* 🧑‍🤝‍🧑 Multi-person emotion tracking
* 🎥 Video recording with emotion logs

---

## 📂 Project Structure

```
emotion-detection/
│── main.py
│── requirements.txt
│── README.md
```

---

## 🙌 Acknowledgements

* OpenCV for computer vision tools
* DeepFace for emotion recognition models

---
