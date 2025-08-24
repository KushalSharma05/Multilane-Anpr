🚗 Multilane ANPR (Automatic Number Plate Recognition)

A real-time **Multilane Automatic Number Plate Recognition (ANPR)** system built using **YOLO** for vehicle & plate detection and **PaddleOCR** for number plate recognition.  
Designed to handle **multiple lanes / video streams simultaneously**, with logging, ROI adjustments, and socket-based integration support.  

## ✨ Features
- 🔍 **YOLO-based detection** – detects vehicles and license plates in real-time.  
- 📝 **PaddleOCR recognition** – extracts text from detected plates.  
- 🎥 **Multilane support** – process multiple streams (CCTV, IP cameras, or video files).  
- 💾 **Logging system** – store recognized plates with timestamps.  
- ⚡ **Socket integration** – connect with external systems (e.g., barriers, parking).  
- 📐 **Configurable ROIs** – adjust detection zones per lane via `config.json`.  

---

## 🛠️ Tech Stack
- [YOLOv5/YOLOv8](https://github.com/ultralytics/yolov5) – object detection.  
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) – text recognition.  
- [OpenCV](https://opencv.org/) – video stream handling.  
- [Python Sockets](https://docs.python.org/3/library/socket.html) – external integration.  

