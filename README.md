# Damaged Car Image Preprocessing Pipeline for Computer Vision using OpenCV

**Team:**

* Renjini L
* Abisha J
* Varsha M S
* Nithi Krishna A D

---

## 📄 Project Description

This repository provides a complete image preprocessing pipeline to prepare damaged car images for downstream computer vision tasks such as damage detection or severity classification. Leveraging OpenCV and Python, the pipeline performs resizing, cropping, color-space conversions, thresholding, contour-based masking, and optional YOLO-based damage localization. A Streamlit web interface allows users to interactively upload images, apply preprocessing steps, visualize results, and download processed outputs.

## 🎯 Key Features

* **Image Resizing & Cropping**: Focus on region of interest with configurable dimensions.
* **Color Space Conversion**: Switch between BGR, RGB, and HSV to suit analysis needs.
* **Thresholding Techniques**: Binary, Adaptive, and Otsu’s thresholding to highlight damaged regions.
* **Contour & Mask Extraction**: Automatically isolate the car body and detect damage via contours.
* **YOLOv8 Integration**: Optional use of a pretrained or custom YOLOv8 model for object-based damage localization.
* **Interactive Streamlit App**: User-friendly interface for uploading images, adjusting parameters, and viewing side-by-side comparisons.
* **Exploratory Data Analysis (EDA)**: Generate image dimension statistics, color histograms, and damage pattern counts.

## 🛠️ Requirements

All dependencies are listed in `requirements.txt`. You can install them with:

```bash
pip install -r requirements.txt
```

```text
# requirements.txt
opencv-python
numpy
matplotlib
pandas
streamlit
ultralytics
onnxruntime
```

## ⚙️ How It Works

1. **Model Setup**:

   * By default, the app loads the `yolov8n.pt` model from Ultralytics.
   * You may upload a custom YOLO model (`.pt`, `.onnx`, or `.yaml`) via the sidebar.

2. **Image Preprocessing Pipeline**:

   * **Resize**: Adjust image dimensions to the specified width and height.
   * **Crop**: Optionally crop to a user-defined bounding box.
   * **Color Spaces**: Convert the processed image to BGR, RGB, and HSV channels.
   * **Thresholds**: Apply Binary, Adaptive, and Otsu’s thresholding to grayscale.

3. **Damage Detection & Localization**:

   * **Contour+Mask**: Extract car mask, find contours of damaged regions, filter by area, and draw bounding boxes.
   * **YOLOv8**: Run inference with a confidence threshold to detect damage classes and draw labeled boxes.

4. **EDA Insights**:

   * Calculate and display image dimension statistics (width, height, aspect ratio).
   * Compute mean and standard deviation of color channels.
   * Plot color histograms and damage pattern counts (bar chart).

5. **Download**:

   * Option to download the processed image directly from the sidebar.

## 🚀 How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/damaged-car-preprocessing.git
   cd damaged-car-preprocessing
   ```
2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Streamlit App**:

   ```bash
   streamlit run app.py
   ```
4. **Access the Web Interface**:

   * Open your browser to `http://localhost:8501`.

## 💡 Typical Usage

1. Start the app and upload a damaged car image (JPG/PNG).
2. In the sidebar, adjust **Resize Width/Height** (e.g., 800×600).
3. Toggle **Enable Crop** to focus on specific regions; set `X`, `Y`, `Width`, and `Height`.
4. Select **Detection Method**:

   * *Contour+Mask* (default): for classic CV-based damage segmentation.
   * *YOLOv8*: for deep-learning-based detection.
5. Set **Min Contour Area** (e.g., 500) or **YOLO Confidence Threshold** (e.g., 0.3).
6. Click **Show EDA Insights** to view statistics and histograms.
7. Download the processed image via the sidebar **Download** button.

---

