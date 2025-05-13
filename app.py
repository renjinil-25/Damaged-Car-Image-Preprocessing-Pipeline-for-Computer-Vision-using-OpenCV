import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Install and import YOLO (Ultralytics YOLOv8)
# pip install ultralytics
from ultralytics import YOLO

# --- Model File Handling ---
def ensure_model_file(uploaded, default='yolov8n.pt'):
    """
    Saves an uploaded file (pt, onnx, or yaml) locally and returns its path,
    otherwise returns the default model path.
    """
    if uploaded:
        os.makedirs('models', exist_ok=True)
        temp_path = os.path.join('models', uploaded.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded.getbuffer())
        return temp_path
    return default

@st.cache_resource
def load_yolo_model(model_path: str):
    """
    Loads a YOLO model from a .pt, .onnx, or .yaml file.
    """
    return YOLO(model_path)

# --- Image Preprocessing Functions ---
def resize_image(img, width=None, height=None):
    h, w = img.shape[:2]
    if width and height:
        return cv2.resize(img, (width, height))
    if width:
        ratio = width / float(w)
        return cv2.resize(img, (width, int(h * ratio)))
    if height:
        ratio = height / float(h)
        return cv2.resize(img, (int(w * ratio), height))
    return img

def crop_image(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def convert_color_spaces(img):
    return {
        'BGR': img,
        'RGB': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        'HSV': cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    }

def apply_thresholds(gray):
    th_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    th_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    _, th_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return {'Binary': th_binary, 'Adaptive': th_adaptive, 'Otsu': th_otsu}

# --- Car Mask Extraction & Contours ---
def extract_car_mask(img, border_gap=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dil = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    valid = []
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if x<=border_gap or y<=border_gap or x+w>=W-border_gap or y+h>=H-border_gap:
            continue
        valid.append(cnt)
    if not valid:
        return np.ones_like(gray, dtype=np.uint8)*255
    car_cnt = max(valid, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [car_cnt], -1, 255, -1)
    return mask

# --- Damage Detection (Contour+Mask) ---
CAR_PARTS = {
    'Front Bumper': (0.15,0.75,0.85,0.98),
    'Hood': (0.15,0.50,0.85,0.75),
    'Windshield': (0.15,0.35,0.85,0.50),
    'Roof': (0.15,0.20,0.85,0.35),
    'Left Door': (0.00,0.35,0.15,0.90),
    'Right Door': (0.85,0.35,1.00,0.90),
    'Rear Bumper': (0.15,0.98,0.85,1.15)
}

def assign_part(x,y,w,h,img_shape):
    H,W = img_shape[:2]
    cx,cy = x+w/2, y+h/2
    for part,(x0,y0,x1,y1) in CAR_PARTS.items():
        if x0*W<=cx<=x1*W and y0*H<=cy<=y1*H:
            return part
    return 'Unknown'

def detect_damage_contours_with_mask(img, min_area=500):
    mask = extract_car_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_car = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(gray_car, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours,_ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes=[]
    for cnt in contours:
        if cv2.contourArea(cnt)<min_area: continue
        x,y,w,h = cv2.boundingRect(cnt)
        bboxes.append((x,y,w,h))
    return bboxes, mask

# --- Damage Detection (YOLO) ---
def detect_damage_with_yolo(img, model, conf_thresh=0.3):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(rgb)[0]
    detections = []
    for *xyxy,conf,cls in results.boxes.data.tolist():
        if conf<conf_thresh: continue
        x1,y1,x2,y2 = map(int,xyxy)
        cname = model.names[int(cls)]
        detections.append({'bbox':(x1,y1,x2-x1,y2-y1),'score':conf,'class_name':cname})
    return detections

# --- EDA Functions ---
def analyze_dimensions(img_list):
    rows=[{'filename':fn,'width':img.shape[1],'height':img.shape[0],'aspect_ratio':round(img.shape[1]/img.shape[0],2)}
           for fn,img in img_list]
    df=pd.DataFrame(rows)
    return df, df.describe()

def compute_color_stats(img):
    chans=cv2.split(img)
    stats={}
    for c,ch in zip(('B','G','R'),chans):
        stats[f'{c}_mean']=round(np.mean(ch),2)
        stats[f'{c}_std']=round(np.std(ch),2)
    return pd.DataFrame([stats])

def plot_color_histograms(img):
    chans=cv2.split(img)
    plt.figure()
    for ch in chans:
        hist=cv2.calcHist([ch],[0],None,[256],[0,256])
        plt.plot(hist)
    plt.title('Color Histogram')
    st.pyplot(plt)

# --- Streamlit App ---
def main():
    st.title('Damaged Car Detection & Analysis')

    # Model Upload (supports .pt, .onnx, .yaml)
    st.sidebar.header('YOLO Model')
    source = st.sidebar.radio('Model Source',['Default pretrained','Upload custom (pt/onnx/yaml)'])
    uploaded_model=None
    if source.startswith('Upload'):
        uploaded_model=st.sidebar.file_uploader('Upload YOLO model',type=['pt','onnx','yaml','yml'])
    model_path=ensure_model_file(uploaded_model)
    yolo_model=load_yolo_model(model_path)

    # Image Upload
    uploaded=st.file_uploader('Upload a car image',type=['jpg','png','jpeg'])
    if not uploaded:
        st.info('Awaiting an image upload.')
        return
    arr=np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
    img=cv2.imdecode(arr,1)

    # Preprocessing Sidebar
    st.sidebar.header('Preprocessing')
    width=st.sidebar.number_input('Resize Width',value=800,step=10)
    height=st.sidebar.number_input('Resize Height',value=600,step=10)
    crop_enable=st.sidebar.checkbox('Enable Crop')
    if crop_enable:
        x=st.sidebar.number_input('Crop X',value=0)
        y=st.sidebar.number_input('Crop Y',value=0)
        w=st.sidebar.number_input('Crop Width',value=img.shape[1])
        h=st.sidebar.number_input('Crop Height',value=img.shape[0])
    method=st.sidebar.radio('Detection Method',['Contour+Mask','YOLOv8'])
    min_area=st.sidebar.number_input('Min Contour Area',value=500,step=50)
    conf_thresh=st.sidebar.slider('YOLO Confidence Threshold',0.0,1.0,0.3)

    img_r=resize_image(img,width,height)
    img_p=crop_image(img_r,x,y,w,h) if crop_enable else img_r

    st.subheader('Original vs Processed')
    st.image([
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
        cv2.cvtColor(img_p,cv2.COLOR_BGR2RGB)
    ],width=300,caption=['Original','Processed'])

    st.subheader('Damage Detection & Localization')
    vis=img_p.copy()
    part_counts={}
    if method=='Contour+Mask':
        bbs,mask=detect_damage_contours_with_mask(img_p,min_area)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,(0,255,0),2)
        for x0,y0,w0,h0 in bbs:
            cv2.rectangle(vis,(x0,y0),(x0+w0,y0+h0),(0,0,255),2)
            part=assign_part(x0,y0,w0,h0,img_p.shape)
            part_counts[part]=part_counts.get(part,0)+1
    else:
        dets=detect_damage_with_yolo(img_p,yolo_model,conf_thresh)
        for d in dets:
            x1,y1,w1,h1=d['bbox']
            cv2.rectangle(vis,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
            cv2.putText(vis,f"{d['class_name']} {d['score']:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            part_counts[d['class_name']]=part_counts.get(d['class_name'],0)+1
    st.image(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB),caption='Detections',width=300)
    st.json(part_counts)

    if st.button('Show EDA Insights'):
        df_dim,stat_dim=analyze_dimensions([('orig',img)])
        st.subheader('Image Dimensions')
        st.dataframe(df_dim)
        st.write(stat_dim)
        st.subheader('Color Channel Statistics')
        st.table(compute_color_stats(img))
        st.write('Color Distribution Histogram:')
        plot_color_histograms(img)
        if method=='Contour+Mask':
            dct={}
            for x0,y0,w0,h0 in detect_damage_contours_with_mask(img_p,min_area)[0]:
                pr=assign_part(x0,y0,w0,h0,img_p.shape)
                dct[pr]=dct.get(pr,0)+1
            st.subheader('Damage Pattern Counts')
            st.bar_chart(pd.DataFrame.from_dict(dct,orient='index',columns=['count']))
        else:
            st.write('Switch to Contour+Mask to see pattern counts.')

    st.sidebar.header('Download')
    if st.sidebar.button('Download Processed Image'):
        _,buf=cv2.imencode('.png',img_p)
        st.sidebar.download_button('Download',buf.tobytes(),'processed.png')

if __name__=='__main__':
    main()
