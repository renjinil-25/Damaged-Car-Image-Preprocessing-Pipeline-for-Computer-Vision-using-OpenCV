import os
import cv2
import time
import numpy as np
import onnxruntime as ort

# Paths
BASE_DIR    = os.path.dirname(__file__)
MODEL_ONNX  = os.path.join(BASE_DIR, "models", "yolov7-nms-640.onnx")
CLASS_NAMES = os.path.join(BASE_DIR, "models", "coco.names")

# Load classes & colors
with open(CLASS_NAMES) as f:
    classes = [c.strip() for c in f]
np.random.seed(42)
colors = np.random.randint(0,255,(len(classes),3),dtype=np.uint8)

# Create ONNX Runtime session
sess = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
# If the model uses dynamic axes, you can ignore shape checks; we'll build the blob directly

def preprocess(frame, size=(640,640)):
    h, w = frame.shape[:2]
    scale = min(size[0]/w, size[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    img = cv2.resize(frame, (nw, nh))
    canvas = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
    dx, dy = (size[0]-nw)//2, (size[1]-nh)//2
    canvas[dy:dy+nh, dx:dx+nw] = img
    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, size, swapRB=True, crop=False)
    return blob, scale, dx, dy

def visualize(frame, outputs, scale, dx, dy):
    # The end2end ONNX typically returns [num_dets, det_boxes, det_scores, det_classes]
    if len(outputs) == 4:
        num, boxes, scores, classes_out = outputs
        num = int(num[0])
        for i in range(num):
            x1,y1,x2,y2 = boxes[0,i]
            cls_id = int(classes_out[0,i])
            conf   = float(scores[0,i])
            color = [int(c) for c in colors[cls_id]]
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            cv2.putText(frame, f"{classes[cls_id]}: {conf:.2f}",
                        (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        # Fallback: single output -> [1, N, 85] raw preds (no in‑graph NMS)
        preds = outputs[0]
        # you can call the same manual NMS postprocess from above
        # ...
        pass

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob, scale, dx, dy = preprocess(frame)
    # ONNX Runtime expects NHWC or NCHW depending on export; here blob is NCHW
    outputs = sess.run(None, {inp_name: blob})
    visualize(frame, outputs, scale, dx, dy)

    cv2.imshow("YOLOv7 ONNXRuntime", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
