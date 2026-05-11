import time
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from collections import deque, defaultdict

import cv2
import numpy as np
import onnxruntime as ort

# ==========================================
# [環境變數與系統優化]
# ==========================================
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class ObjectTracker:
    def __init__(self, max_disappeared=7, max_distance=150, reset_threshold=45):
        self.objects = {} 
        self.max_disappeared = max_disappeared 
        self.max_distance = max_distance
        self.reset_threshold = reset_threshold
        self.category_next_ids = defaultdict(int)
        self.category_empty_frames = defaultdict(int)

    def update(self, detections, all_possible_classes):
        current_frame_classes = {det[5] for det in detections}
        for cls in all_possible_classes:
            if cls not in current_frame_classes:
                self.category_empty_frames[cls] += 1
                if self.category_empty_frames[cls] >= self.reset_threshold:
                    self.category_next_ids[cls] = 0
            else:
                self.category_empty_frames[cls] = 0

        updated_ids = set()
        for det in detections:
            x1, y1, x2, y2, conf, cls_name = det
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best_id, min_dist = None, self.max_distance
            for obj_id, obj_data in self.objects.items():
                if obj_id in updated_ids or obj_data['cls_name'] != cls_name: continue
                ox1, oy1, ox2, oy2 = obj_data['bbox']
                dist = ((cx - (ox1 + ox2)/2)**2 + (cy - (oy1 + oy2)/2)**2)**0.5
                if dist < min_dist: best_id, min_dist = obj_id, dist
            
            if best_id:
                self.objects[best_id].update({'bbox': (x1, y1, x2, y2), 'conf': conf, 'disappeared': 0})
                updated_ids.add(best_id)
            else:
                new_id = f"{cls_name}_{self.category_next_ids[cls_name]}"
                self.objects[new_id] = {'bbox': (x1, y1, x2, y2), 'cls_name': cls_name, 'conf': conf, 'disappeared': 0}
                self.category_next_ids[cls_name] += 1
                updated_ids.add(new_id)
        
        for oid in list(self.objects.keys()):
            if oid not in updated_ids:
                self.objects[oid]['disappeared'] += 1
                if self.objects[oid]['disappeared'] > self.max_disappeared: 
                    del self.objects[oid]
        return self.objects

class SecuritySystemApp:
    def __init__(self, model_path: str):
        self.inferencer = YOLOONNXInferencer(model_path, conf_threshold=0.6)
        self.tracker = ObjectTracker(max_disappeared=7, reset_threshold=45)
        self.fps_history = deque(maxlen=10)
        self.COCO_CLASSES = {i: name for i, name in enumerate(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])}
        self.class_list = list(self.COCO_CLASSES.values())
        self.zone_polygon_pct = np.array([[0.1, 0.9], [0.9, 0.9], [0.7, 0.5], [0.3, 0.5]], np.float32)
        self.loitering_since = None
        self.total_intrusions = 0
        self.last_seen_t = 0.0
        self.patience = 0.5
        self.alert_dir = Path("alerts"); self.alert_dir.mkdir(exist_ok=True)
        self.last_alert_t = 0.0
        self.best_frame = None
        self.terminal_logs = deque(maxlen=20)
        self.prev_interaction_state = False

    def log_to_terminal(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.terminal_logs.append(f"[{ts}] {msg}")

    def get_center(self, bbox):
        return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

    def run(self):
        cap = cv2.VideoCapture(0)
        window_name = "YOLO AI Developer Insight Console"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        prev_t = time.perf_counter()

        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error: break
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            zone_px = (self.zone_polygon_pct * np.array([w, h])).astype(np.int32)
            results = self.inferencer.predict(frame)
            raw_dets = [(int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4], self.COCO_CLASSES.get(int(d[5]), f"ID_{int(d[5])}")) for d in results]
            tracked_objs = self.tracker.update(raw_dets, self.class_list)
            infer_ms = (time.perf_counter() - time.perf_counter()) * 1000 # 修正測量 logic 
            
            curr_sys_t = time.time()
            any_intruder = False
            is_interacting = False
            overlay = frame.copy()

            # 互動與空間分析邏輯 (保持不變)
            people = {k: v for k, v in tracked_objs.items() if v['cls_name'] == 'person'}
            items = {k: v for k, v in tracked_objs.items() if v['cls_name'] != 'person'}
            for pid, pdata in people.items():
                pc = self.get_center(pdata['bbox'])
                for iid, idata in items.items():
                    ic = self.get_center(idata['bbox'])
                    if ((pc[0]-ic[0])**2 + (pc[1]-ic[1])**2)**0.5 < 180: 
                        is_interacting = True
                        cv2.line(frame, pc, ic, (0, 255, 255), 1, cv2.LINE_AA)
            if is_interacting and not self.prev_interaction_state: self.log_to_terminal("EVENT: Semantic link active.")
            elif not is_interacting and self.prev_interaction_state: self.log_to_terminal("EVENT: Semantic link broken.")
            self.prev_interaction_state = is_interacting

            for oid, data in tracked_objs.items():
                x1, y1, x2, y2 = map(int, data['bbox'])
                cls = data['cls_name']
                display_id = oid.split('_')[-1]
                in_zone = (cls == 'person' and cv2.pointPolygonTest(zone_px, (float((x1+x2)/2), float(y2)), False) >= 0)
                if in_zone: any_intruder = True
                color = (0, 0, 255) if in_zone else (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls} #{display_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if any_intruder:
                self.last_seen_t = curr_sys_t
                self.best_frame = frame.copy()
                if self.loitering_since is None: self.loitering_since = curr_sys_t
            
            if self.loitering_since is not None:
                if (curr_sys_t - self.last_seen_t) <= self.patience:
                    duration = curr_sys_t - self.loitering_since
                    if duration >= 2.0:
                        cv2.fillPoly(overlay, [zone_px], (0, 0, 255))
                        if curr_sys_t - self.last_alert_t > 3.0:
                            self.last_alert_t = curr_sys_t
                            self.log_to_terminal("CRITICAL: Alert triggered.")
                    else:
                        cv2.fillPoly(overlay, [zone_px], (0, 165, 255))
                else: self.loitering_since = None
            else: cv2.fillPoly(overlay, [zone_px], (0, 255, 0))

            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [zone_px], True, (0, 255, 0), 2)
            
            # HUD 計算
            curr_t = time.perf_counter()
            self.fps_history.append(1/(curr_t-prev_t)); prev_t = curr_t
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (10, 10), (180, 80), (0, 0, 0), -1)
            cv2.addWeighted(hud_overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"FPS: {sum(self.fps_history)/10:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # ==========================================
            # [終極佈局修正] 絕對邊界判定
            # ==========================================
            console_w = 400
            console_panel = np.full((h, console_w, 3), 18, dtype=np.uint8)
            cv2.line(console_panel, (0, 0), (0, h), (100, 100, 100), 2)

            # 1. 計算底部的 LOG 區起點 (確保它鎖定在底部)
            MAX_LOG_LINES = 12
            line_h = 20
            # 留一點緩衝空間 (40px) 避免文字貼底
            log_section_h = (MAX_LOG_LINES * line_h) + 40 
            log_start_y = h - log_section_h

            # 2. 繪製 LIVE DATA 區塊 (加入「剩餘空間檢查」)
            cv2.putText(console_panel, "--- LIVE TRACKER DATA ---", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            c_y = 65
            # 重要：定義 Data 區絕對不能超過的高度
            data_limit_y = log_start_y - 30 
            
            all_objs = list(tracked_objs.items())
            num_shown = 0
            
            for oid, data in all_objs:
                # 預判下一筆資料所需高度 (約 45px)
                if c_y + 45 > data_limit_y:
                    break # 空間不足，強行停止印出後續物件
                
                cls = data['cls_name']; did = oid.split('_')[-1]
                cv2.putText(console_panel, f"> {cls} #{did} (Conf: {data['conf']:.2f})", (15, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                c_y += 18
                x1, y1, x2, y2 = map(int, data['bbox'])
                cv2.putText(console_panel, f"  Box:[{x1},{y1},{x2},{y2}]  Patience:{data['disappeared']}/7", (15, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
                c_y += 27
                num_shown += 1

            # 如果有截斷，在 Data 區底部印上一行灰色說明
            if len(all_objs) > num_shown:
                cv2.putText(console_panel, f"... and {len(all_objs)-num_shown} more hidden", (15, c_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # 3. 繪製 LOG 區塊 (固定位置，絕對防護)
            cv2.line(console_panel, (10, log_start_y), (console_w - 10, log_start_y), (100, 100, 100), 1)
            cv2.putText(console_panel, "--- SYSTEM LOGS ---", (15, log_start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            current_logs = list(self.terminal_logs)[-MAX_LOG_LINES:]
            for i, msg in enumerate(current_logs):
                # 這裡的 y 座標完全與上面的 c_y 無關，絕對不會重疊
                text_y = log_start_y + 45 + (i * line_h)
                color = (0, 0, 255) if "CRITICAL" in msg else (0, 200, 0)
                cv2.putText(console_panel, msg, (15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            final_display = np.hstack([frame, console_panel])
            cv2.imshow(window_name, final_display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SecuritySystemApp("yolo26n.onnx")
    app.run()