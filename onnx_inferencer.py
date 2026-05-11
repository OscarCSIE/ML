import time
import os
import logging
from pathlib import Path
from typing import Tuple, Dict
from collections import deque, defaultdict

import cv2
import numpy as np
import onnxruntime as ort

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# [基礎模型類別維持不變]
class YOLOONNXInferencer:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        if not self.model_path.exists(): raise FileNotFoundError(f"找不到模型：{self.model_path}")
        self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = np.expand_dims(np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
        predictions = self.session.run(None, {self.input_name: img_tensor})[0][0]
        valid_preds = predictions[predictions[:, 4] > self.conf_threshold]
        if len(valid_preds) > 0:
            valid_preds[:, [0, 2]] *= (w / 640)
            valid_preds[:, [1, 3]] *= (h / 640)
        return valid_preds

# ==========================================
# [優化版] 類別獨立重製之多目標追蹤器
# ==========================================
class ObjectTracker:
    def __init__(self, max_disappeared=7, reset_threshold=45):
        self.objects = {} 
        self.max_disappeared = max_disappeared 
        
        # 使用字典為每個類別獨立計數
        self.category_next_ids = defaultdict(int)     # 各類別下一個可用 ID
        self.category_empty_frames = defaultdict(int) # 各類別連續消失的幀數
        self.reset_threshold = reset_threshold        # 某類別消失多久後重置該類 ID

    def update(self, detections, all_possible_classes):
        # 1. 更新所有已知類別的消失計數
        current_frame_classes = {det[5] for det in detections}
        for cls in all_possible_classes:
            if cls in current_frame_classes:
                self.category_empty_frames[cls] = 0
            else:
                self.category_empty_frames[cls] += 1
                # 如果該類別消失過久，重置該類別的編號計數
                if self.category_empty_frames[cls] >= self.reset_threshold:
                    if self.category_next_ids[cls] != 0:
                        logging.info(f"類別 [{cls}] 已長時間未出現，編號重置。")
                    self.category_next_ids[cls] = 0

        updated_ids = set()
        for det in detections:
            x1, y1, x2, y2, conf, cls_name = det
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            best_id, min_dist = None, 150
            for obj_id, obj_data in self.objects.items():
                if obj_id in updated_ids or obj_data['cls_name'] != cls_name: continue
                ox1, oy1, ox2, oy2 = obj_data['bbox']
                dist = ((cx - (ox1 + ox2)/2)**2 + (cy - (oy1 + oy2)/2)**2) ** 0.5
                if dist < min_dist: best_id, min_dist = obj_id, dist
                    
            if best_id is not None:
                self.objects[best_id].update({'bbox': (x1, y1, x2, y2), 'conf': conf, 'disappeared': 0})
                updated_ids.add(best_id)
            else:
                # 取得該類別專屬的下一個 ID (例如 person #0, bottle #0)
                new_id = f"{cls_name}_{self.category_next_ids[cls_name]}"
                self.objects[new_id] = {'bbox': (x1, y1, x2, y2), 'cls_name': cls_name, 'conf': conf, 'disappeared': 0}
                self.category_next_ids[cls_name] += 1
                updated_ids.add(new_id)
                
        for obj_id in list(self.objects.keys()):
            if obj_id not in updated_ids:
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared: del self.objects[obj_id]
        return self.objects

if __name__ == "__main__":
    try:
        inferencer = YOLOONNXInferencer("yolo26n.onnx", conf_threshold=0.5)
        # 設定消失 45 幀 (約 2 秒) 後重置該類別編號
        tracker = ObjectTracker(max_disappeared=7, reset_threshold=45)
        
        COCO_CLASSES = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        class_list = list(COCO_CLASSES.values())
        
        cap = cv2.VideoCapture(0)
        window_name = "YOLO26 Category-Specific Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        fps_history = deque(maxlen=10)
        prev_time = time.perf_counter()
        
        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error: break
                
            ret, frame = cap.read()
            if not ret: break
            
            start_time = time.perf_counter()
            results = inferencer.predict(frame)
            
            raw_dets = []
            for d in results:
                cls_name = COCO_CLASSES.get(int(d[5]), f"ID_{int(d[5])}")
                raw_dets.append((int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4], cls_name))
            
            # 傳入當前影格結果與所有可能類別
            tracked_objs = tracker.update(raw_dets, class_list)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            for obj_id, data in tracked_objs.items():
                x1, y1, x2, y2 = map(int, data['bbox'])
                # 只顯示數字部分，隱藏內部的類別前綴
                display_id = obj_id.split('_')[-1]
                label = f"{data['cls_name']} #{display_id} {data['conf']:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # HUD 渲染
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (10, 10), (160, 60), (0, 0, 0), -1)
            cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0, frame)
            curr_time = time.perf_counter()
            fps_history.append(1 / (curr_time - prev_time)); prev_time = curr_time
            cv2.putText(frame, f"FPS: {sum(fps_history)/len(fps_history):.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Infer: {inference_time:.1f}ms", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    except Exception as e: logging.error(f"執行異常: {e}")
    finally:
        if 'cap' in locals(): cap.release()
        cv2.destroyAllWindows()