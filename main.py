import cv2
from ultralytics import YOLO
import os
import psutil
import logging
import time
import gc
import sys
from PIL import Image
import imagehash
from datetime import datetime
from paddleocr import PaddleOCR
import socket
import json
import numpy as np
import threading
from collections import deque
import queue
import math
import re # Added for potential URL parsing

# Utility function for resource paths
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Directory helpers
def get_exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

base_dir = get_exe_dir()

# Global logging setup
global_log_dir = os.path.join(base_dir, "global_logs")
os.makedirs(global_log_dir, exist_ok=True)
global_log_file = os.path.join(global_log_dir, "global_detection.log")
global_file_handler = logging.FileHandler(global_log_file, encoding="utf-8")
global_console_handler = logging.StreamHandler()
global_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
global_file_handler.setFormatter(global_formatter)
global_console_handler.setFormatter(global_formatter)
global_logger = logging.getLogger("global")
global_logger.setLevel(logging.INFO)
global_logger.addHandler(global_file_handler)
global_logger.addHandler(global_console_handler)

def global_log(message):
    global_logger.info(message)

# Function to clean old logs
def clean_old_logs(directory, days=30):
    now = time.time()
    cutoff = now - (days * 86400)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
            try:
                os.remove(filepath)
                global_log(f"🗑️ Deleted old log file: {filepath}")
            except Exception as e:
                global_log(f"❌ Failed to delete old log file {filepath}: {e}")

# Clean global logs at startup
clean_old_logs(global_log_dir)

# Configuration loading
config_path = os.path.join(base_dir, "config1.json")
if not os.path.exists(config_path):
    sys.exit("❌ config.json file not found. Please create with appropriate fields.")

def load_config(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"❌ Error parsing config.json: {e}")

config = load_config(config_path)
config_last_modified = os.path.getmtime(config_path)

# Global configurations
global_config = config.get("global", {})
heatmap_threshold = global_config.get("heatmap_threshold", 1000)
frame_skip = global_config.get("frame_skip", 5)
distance_threshold = global_config.get("distance_threshold", 50)
min_contour_area = global_config.get("min_contour_area", 500)
vehicle_threshold_time = global_config.get("vehicle_threshold_time", 0.5)
DEDUP_COOLDOWN = global_config.get("deduplication_cooldown", 15)
display_window_active = global_config.get("display_window", True)
DISPLAY_WIDTH = global_config.get("display_width", 960)
DISPLAY_HEIGHT = global_config.get("display_height", 540)
max_frame_failures = global_config.get("max_frame_failures", 10)

# Lanes configuration
lanes = config.get("lanes", [])
if not lanes:
    sys.exit("❌ No lanes defined in config.json.")

# Model loading (global, shared)
try:
    model_path = resource_path("best.pt")
    model = YOLO(model_path)
    global_log("✅ YOLOv8 model loaded successfully.")
except Exception as e:
    global_log(f"❌ Error loading YOLOv8 model: {e}")
    sys.exit()

try:
    ocr_model_dir = resource_path("best_model")
    ocr = PaddleOCR(
        det=True,
        det_model_dir=resource_path("det/en/en_PP-OCRv3_det_infer"),
        cls_model_dir=resource_path("cls/ch_ppocr_mobile_v2.0_cls_infer"),
        rec=True,
        use_angle_cls=True,
        use_gpu=False,
        rec_model_dir=ocr_model_dir,
        type='ocr',
        lang='en'
    )
    global_log("✅ PaddleOCR loaded successfully.")
except Exception as e:
    global_log(f"❌ Error loading PaddleOCR: {e}. Check model paths and install.")
    sys.exit()

# Global locks for shared models
yolo_lock = threading.Lock()
ocr_lock = threading.Lock()

# Global display queue
display_queue = queue.Queue()

# Global shutdown event for all lanes
global_shutdown_event = threading.Event()

# Per-lane data structure
lane_data = {}

def parse_lane_id_from_url(url):
    # Parse lane number as the last digit before the port colon
    match = re.search(r'(\d+\.\d+\.\d+\.\d+)', url)
    if match:
        ip = match.group(1)  # e.g., "192.168.17.181"
        last_octet = ip.split('.')[-1]  # "181"
        return int(last_octet)
    return None

for lane_config in lanes:
    lane_id = lane_config.get("lane_id")
    stream_url = lane_config.get("source", "").strip()

    # If lane_id is missing or empty, try parsing it from URL
    if not lane_id:
        lane_id = parse_lane_id_from_url(stream_url)
        if not lane_id:
            sys.exit(f"❌ Could not parse lane_id from URL: {stream_url}")

        lane_id = int(lane_id)

        # Save parsed lane_id back to config.json
        try:
            with open(config_path, "r+", encoding="utf-8") as f:
                full_config = json.load(f)
                updated = False
                for lconfig in full_config.get("lanes", []):
                    if lconfig.get("source", "").strip() == stream_url:
                        lconfig["lane_id"] = lane_id
                        updated = True
                        break
                if updated:
                    f.seek(0)
                    json.dump(full_config, f, indent=2)
                    f.truncate()
                    global_log(f"✅ Parsed lane_id {lane_id} saved to config.json for source: {stream_url}")
                else:
                    global_log(f"⚠️ No matching lane found in config.json for source: {stream_url}")
        except Exception as e:
            global_log(f"❌ Failed to save parsed lane_id {lane_id} to config.json: {e}")

    else:
        lane_id = int(lane_id)  # Ensure integer
        global_log(f"ℹ️ Using existing lane_id {lane_id} from config.json for source: {stream_url}")

    server_ip = lane_config.get("server_ip", "").strip()
    server_port = lane_config.get("server_port")
    use_socket = False
    if server_ip and str(server_port).isdigit():
        server_port = int(server_port)
        use_socket = True

    # Per-lane directories
    lane_base_dir = os.path.join(base_dir, f"lane_{lane_id}")
    log_dir = os.path.join(lane_base_dir, "logs")
    image_output_dir = os.path.join(lane_base_dir, "detected_plates")
    text_output_dir = os.path.join(lane_base_dir, "detected_numplate_text")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)

    # Clean per-lane logs at startup
    clean_old_logs(log_dir)

    # Per-lane logging
    log_file = os.path.join(log_dir, "detection.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    lane_logger = logging.getLogger(f"lane_{lane_id}")
    lane_logger.setLevel(logging.INFO)
    lane_logger.addHandler(file_handler)
    lane_logger.addHandler(console_handler)

    plate_log_file = os.path.join(log_dir, "plate_detections.log")
    plate_file_handler = logging.FileHandler(plate_log_file, encoding="utf-8")
    plate_formatter = logging.Formatter("%(asctime)s - %(message)s")
    plate_file_handler.setFormatter(plate_formatter)
    plate_logger = logging.getLogger(f"plate_lane_{lane_id}")
    plate_logger.setLevel(logging.INFO)
    plate_logger.addHandler(plate_file_handler)

    # Per-lane ROI and other vars
    roi_dict = lane_config.get("roi", {})
    x1, y1, x2, y2 = 0, 0, 0, 0
    roi_active = False
    roi_auto_set = False
    initial_roi_dynamically_set = False
    fixed_roi_width = lane_config.get("fixed_roi_width", 640)
    fixed_roi_height = lane_config.get("fixed_roi_height", 360)
    detection_counter = 0
    plate_coords = []
    heatmap = None
    roi_changed_flag = False

    # Per-lane display flag
    display_active = lane_config.get("display", True)

    # Per-lane queues and events
    image_queue = deque(maxlen=10)
    ocr_queue = queue.Queue() # Per-lane OCR queue for simplicity, but we'll use workers
    shutdown_event = threading.Event()
    client_socket = None
    socket_lock = threading.Lock()

    lane_data[lane_id] = {
        "config": lane_config,
        "stream_url": stream_url,
        "server_ip": server_ip,
        "server_port": server_port,
        "use_socket": use_socket,
        "lane_base_dir": lane_base_dir,
        "log_dir": log_dir,
        "image_output_dir": image_output_dir,
        "text_output_dir": text_output_dir,
        "logger": lane_logger,
        "plate_logger": plate_logger,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "roi_active": roi_active,
        "roi_auto_set": roi_auto_set,
        "initial_roi_dynamically_set": initial_roi_dynamically_set,
        "detection_counter": detection_counter,
        "plate_coords": plate_coords,
        "heatmap": heatmap,
        "roi_changed_flag": roi_changed_flag,
        "image_queue": image_queue,
        "ocr_queue": ocr_queue,
        "shutdown_event": shutdown_event,
        "client_socket": client_socket,
        "socket_lock": socket_lock,
        "fixed_roi_width": fixed_roi_width,
        "fixed_roi_height": fixed_roi_height,
        "plate_buffer": deque(maxlen=30),
        "session_id": 0,
        "plate_detected_in_this_frame": False,
        "frame_count": 0,
        "last_ocr_save_time": time.time(),
        "frame_failures": 0,
        "last_detected_plate_coords": None,
        "last_vehicle_leaving_time": time.time(),
        "prev_gray_frame": None,
        "recently_saved_plates": {},
        "roi_ratio": lane_config.get("roi_ratio", {"min": 0.25, "max": 0.75}),
        "display_active": display_active
    }

# Utility functions (adapted for per-lane)
def log(lane_id, message):
    if lane_id == 0:
        global_log(message)
    else:
        lane_data[lane_id]["logger"].info(message)

def log_plate_detection(lane_id, plate_text):
    lane_data[lane_id]["plate_logger"].info(plate_text)

def get_sharpness(img):
    try:
        if img is None or img.size == 0 or img.ndim not in [2, 3]:
            return 0.0
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        return 0.0

def log_cpu_mem():
    process = psutil.Process()
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_info().rss / (1024 * 1024)
    global_log(f"📊 Global CPU: {cpu:.2f}% | Memory: {mem:.2f} MB")

def save_roi_to_config(lane_id, x1, y1, x2, y2, config_path):
    try:
        with open(config_path, "r") as f:
            full_config = json.load(f)
        for lconfig in full_config["lanes"]:
            if lconfig["lane_id"] == lane_id:
                lconfig["roi"] = {
                    "x1": str(int(x1)),
                    "y1": str(int(y1)),
                    "x2": str(int(x2)),
                    "y2": str(int(y2)),
                }
                if "initial_roi" in lconfig:
                    del lconfig["initial_roi"]
                break
        with open(config_path, "w") as f:
            json.dump(full_config, f, indent=2)
        log(lane_id, f"📁 Final ROI saved to {config_path} for lane {lane_id}")
    except Exception as e:
        log(lane_id, f"❌ Failed to save ROI to config for lane {lane_id}: {e}")

def calculate_roi(lane_id, frame_shape):
    data = lane_data[lane_id]
    if len(data["plate_coords"]) < heatmap_threshold:
        return False
    h, w = frame_shape[:2]
    if data["heatmap"] is None:
        data["heatmap"] = np.zeros((h, w), dtype=np.float32)

    for x_min_coord, y_min_coord, x_max_coord, y_max_coord in data["plate_coords"]:
        x_min_clip = max(0, x_min_coord)
        y_min_clip = max(0, y_min_coord)
        x_max_clip = min(w, x_max_coord)
        y_max_clip = min(h, y_max_coord)
        if x_min_clip < x_max_clip and y_min_clip < y_max_clip:
            data["heatmap"][y_min_clip:y_max_clip, x_min_clip:x_max_clip] += 1

    min_heatmap_value = heatmap_threshold * 0.5
    high_density = data["heatmap"] > min_heatmap_value
    if not np.any(high_density):
        log(lane_id, "⚠️ No high-density region found in heatmap.")
        return False
    y_indices, x_indices = np.where(high_density)

    center_x, center_y = int(np.mean(x_indices)), int(np.mean(y_indices))

    half_width = data["fixed_roi_width"] // 2
    half_height = data["fixed_roi_height"] // 2

    new_x1 = max(0, center_x - half_width)
    new_y1 = max(0, center_y - half_height)
    new_x2 = min(w, center_x + half_width)
    new_y2 = min(h, center_y + half_height)

    achieved_width, achieved_height = new_x2 - new_x1, new_y2 - new_y1

    if achieved_width < data["fixed_roi_width"] * 0.9 or achieved_height < data["fixed_roi_height"] * 0.9:
        log(lane_id, f"⚠️ Calculated ROI smaller than fixed size ({achieved_width}x{achieved_height} vs {data['fixed_roi_width']}x{data['fixed_roi_height']}). Collecting more detections.")
        return False

    if not (data["x1"] == new_x1 and data["y1"] == new_y1 and data["x2"] == new_x2 and data["y2"] == new_y2):
        data["x1"], data["y1"], data["x2"], data["y2"] = new_x1, new_y1, new_x2, new_y2
        data["roi_active"], data["roi_auto_set"] = True, True
        log(lane_id, f"✅ Heatmap-based ROI set and finalized: ({new_x1}, {new_y1}) to ({new_x2}, {new_y2})")
        save_roi_to_config(lane_id, new_x1, new_y1, new_x2, new_y2, config_path)
        data["roi_changed_flag"] = True
    else:
        log(lane_id, "ℹ️ Heatmap-based ROI calculated but did not change current ROI.")

    return True

# Socket functions per lane
def connect_to_server(lane_id):
    data = lane_data[lane_id]
    with data["socket_lock"]:
        try:
            if data["client_socket"]:
                data["client_socket"].shutdown(socket.SHUT_RDWR)
                data["client_socket"].close()
            data["client_socket"] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data["client_socket"].settimeout(5)
            data["client_socket"].connect((data["server_ip"], data["server_port"]))
            log(lane_id, f"🔗 Connected to server at {data['server_ip']}:{data['server_port']} for lane {lane_id}")
            return True
        except Exception as e:
            log(lane_id, f"❌ Could not connect to server for lane {lane_id}: {e}")
            data["client_socket"] = None
            return False

def send_plate_over_socket(lane_id, image_bytes, plate_number, filename):
    data = lane_data[lane_id]
    with data["socket_lock"]:
        if not data["client_socket"]:
            log(lane_id, f"❌ Socket not connected for lane {lane_id}. Cannot send {filename}.")
            threading.Thread(target=connect_to_server, args=(lane_id,)).start()
            return False
        metadata = {
            "Name": filename,
            "Type": "jpeg",
            "Size": len(image_bytes),
            "PlateText": plate_number,
        }
        json_payload = json.dumps(metadata).encode("utf-8")
        try:
            data["client_socket"].sendall(len(json_payload).to_bytes(4, "big"))
            data["client_socket"].sendall(json_payload)
            data["client_socket"].sendall(image_bytes)
            return True
        except Exception as e:
            log(lane_id, f"❌ Send failed for {filename} in lane {lane_id}: {e}. Reconnecting.")
            data["client_socket"] = None
            threading.Thread(target=connect_to_server, args=(lane_id,)).start()
            return False

# Function to reload config
def reload_config():
    global config, config_last_modified, display_window_active
    try:
        new_config = load_config(config_path)
        for lane_config in new_config.get("lanes", []):
            lane_id = lane_config.get("lane_id")
            stream_url = lane_config.get("source", "").strip()
            if not lane_id:
                lane_id = parse_lane_id_from_url(stream_url)
            if lane_id and lane_id in lane_data:
                old_display = lane_data[lane_id]["display_active"]
                new_display = lane_config.get("display", True)
                lane_data[lane_id]["display_active"] = new_display
                if old_display and not new_display:
                    global_log(f"🖼️ Display removed for lane {lane_id} due to config change.")
                elif not old_display and new_display:
                    global_log(f"🖼️ Display added for lane {lane_id} due to config change.")
        display_window_active = new_config.get("global", {}).get("display_window", True)
        config = new_config
        config_last_modified = os.path.getmtime(config_path)
        global_log("✅ Config reloaded.")
    except Exception as e:
        global_log(f"❌ Error reloading config: {e}")

# OCR worker (global pool, process per-lane queues)
def ocr_worker(lane_id):
    data = lane_data[lane_id]
    while not data["shutdown_event"].is_set():
        try:
            current_time = time.time()
            old_plates = [plate for plate, ts in data["recently_saved_plates"].items() if current_time - ts > DEDUP_COOLDOWN]
            for plate in old_plates:
                if plate in data["recently_saved_plates"]:
                    del data["recently_saved_plates"][plate]

            sharpest, coords, sharp_val, frame_shape = data["ocr_queue"].get(timeout=0.5)

            if sharp_val <= 40:
                log(lane_id, f"⏩ Skipping OCR for plate with low sharpness: {sharp_val:.2f} in lane {lane_id}")
                continue

            with ocr_lock:
                result = ocr.ocr(sharpest, cls=True)
                extracted_text = []
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) > 1 and line[1] and len(line[1]) > 0:
                            extracted_text.append(line[1][0])
                plate_text = "".join(extracted_text).strip().replace(" ", "").replace("/", "").upper()
                if not plate_text:
                    plate_text = "UNKNOWN"
                    log(lane_id, "⚠️ No text extracted from plate, saving as UNKNOWN in lane {lane_id}.")

            if plate_text in data["recently_saved_plates"]:
                log(lane_id, f"➡️ Duplicate plate text '{plate_text}' detected within {DEDUP_COOLDOWN}s in lane {lane_id}. Skipping.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}_{plate_text}"
            filename_image = f"{filename_base}.jpg"
            _, buffer = cv2.imencode(".jpg", sharpest)
            image_bytes = buffer.tobytes()

            if data["use_socket"]:
                send_plate_over_socket(lane_id, image_bytes, plate_text, filename_image)

            plate_path = os.path.join(data["image_output_dir"], filename_image)
            cv2.imwrite(plate_path, sharpest)
            text_filename = os.path.join(data["text_output_dir"], f"{filename_base}.txt")
            with open(text_filename, "w") as f:
                f.write(plate_text)

            log(lane_id, f"✅ Plate saved: {plate_path} | Sharpness: {sharp_val:.2f} | Text: {plate_text} in lane {lane_id}")
            log(lane_id, f"📝 Text saved: {text_filename} | Content: {plate_text} in lane {lane_id}")
            log_plate_detection(lane_id, plate_text)
            data["recently_saved_plates"][plate_text] = time.time()

            if not data["roi_auto_set"]:
                data["plate_coords"].append(coords)
                data["detection_counter"] += 1
                log(lane_id, f"🔍 Distinct plate saved for ROI: {data['detection_counter']}/{heatmap_threshold} in lane {lane_id}")
                if data["detection_counter"] >= heatmap_threshold:
                    calculate_roi(lane_id, frame_shape)

            data["session_id"] += 1
        except queue.Empty:
            continue
        except Exception as e:
            log(lane_id, f"❌ Unexpected error in OCR worker for lane {lane_id}: {e}")

# Stream capture thread per lane
def stream_loop(lane_id):
    data = lane_data[lane_id]
    cap = cv2.VideoCapture(data["stream_url"], cv2.CAP_FFMPEG)
    frame_failures_thread = 0
    while not data["shutdown_event"].is_set():
        if not cap.isOpened():
            log(lane_id, f"🔄 Stream not opened for lane {lane_id}. Reinitializing...")
            cap.release()
            cap = cv2.VideoCapture(data["stream_url"], cv2.CAP_FFMPEG)
            time.sleep(1)
            continue
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            if len(data["image_queue"]) == data["image_queue"].maxlen:
                data["image_queue"].popleft()
            data["image_queue"].append(frame)
            frame_failures_thread = 0
        else:
            frame_failures_thread += 1
            log(lane_id, f"⚠️ Frame read failed in thread for lane {lane_id} ({frame_failures_thread}/{max_frame_failures}). Reinitializing...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(data["stream_url"], cv2.CAP_FFMPEG)
            if frame_failures_thread >= max_frame_failures:
                log(lane_id, f"❌ Too many frame read failures for lane {lane_id}. Exiting thread.")
                data["shutdown_event"].set()
                break
        time.sleep(0.001)
    cap.release()
    log(lane_id, f"✅ Stream capture thread stopped for lane {lane_id}.")

# Processing thread per lane
def processing_loop(lane_id):
    data = lane_data[lane_id]
    data["frame_count"] = 0
    while not data["shutdown_event"].is_set():
        if data["image_queue"]:
            try:
                current_frame = data["image_queue"].popleft()
            except IndexError:
                time.sleep(0.005)
                continue
        else:
            time.sleep(0.005)
            continue

        data["frame_count"] += 1
        start_time = time.time()

        process_frame = current_frame.copy()

        # Dynamic initial ROI
        if not data["roi_active"] and not data["roi_auto_set"] and not data["initial_roi_dynamically_set"]:
            height, width = current_frame.shape[:2]
            min_ratio = data["roi_ratio"].get("min", 0.25)
            max_ratio = data["roi_ratio"].get("max", 0.75)
            data["x1"] = int(width * min_ratio)
            data["y1"] = int(height * min_ratio)
            data["x2"] = int(width * max_ratio)
            data["y2"] = int(height * max_ratio)
            data["roi_active"] = True
            data["initial_roi_dynamically_set"] = True
            log(lane_id, f"📥 Dynamically set initial ROI to: ({data['x1']}, {data['y1']}) to ({data['x2']}, {data['y2']}) for lane {lane_id}.")
            data["roi_changed_flag"] = True

        h_frame, w_frame, _ = current_frame.shape
        effective_x1 = max(0, data["x1"])
        effective_y1 = max(0, data["y1"])
        effective_x2 = min(w_frame, data["x2"])
        effective_y2 = min(h_frame, data["y2"])

        if data["roi_changed_flag"]:
            data["prev_gray_frame"] = None
            data["roi_changed_flag"] = False

        data["plate_detected_in_this_frame"] = False
        boxes = []
        trigger_yolo = False

        if data["roi_active"] and effective_x2 > effective_x1 and effective_y2 > effective_y1:
            roi_frame_for_yolo = current_frame[effective_y1:effective_y2, effective_x1:effective_x2].copy()
        else:
            effective_x1, effective_y1, effective_x2, effective_y2 = 0, 0, w_frame, h_frame
            roi_frame_for_yolo = current_frame.copy()

        if roi_frame_for_yolo.size > 0:
            current_gray_frame = cv2.cvtColor(roi_frame_for_yolo, cv2.COLOR_BGR2GRAY)
            current_gray_frame = cv2.GaussianBlur(current_gray_frame, (21, 21), 0)

            if data["prev_gray_frame"] is None:
                data["prev_gray_frame"] = current_gray_frame
                log(lane_id, f"ℹ️ Initializing or resetting prev_gray_frame for motion detection in lane {lane_id}.")

            if data["prev_gray_frame"].shape != current_gray_frame.shape:
                log(lane_id, f"⚠️ Mismatch in prev_gray_frame and current_gray_frame shapes for lane {lane_id}. Resetting.")
                data["prev_gray_frame"] = current_gray_frame
            else:
                diff = cv2.absdiff(data["prev_gray_frame"], current_gray_frame)
                thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    roi_h, roi_w = roi_frame_for_yolo.shape[:2]
                    roi_cx = roi_w // 2
                    roi_cy = roi_h // 2

                    for cnt in contours:
                        if cv2.contourArea(cnt) > min_contour_area:
                            (x_rel, y_rel, w_rel, h_rel) = cv2.boundingRect(cnt)
                            object_cx_rel = x_rel + w_rel // 2
                            object_cy_rel = y_rel + h_rel // 2
                            distance = math.sqrt((object_cx_rel - roi_cx)**2 + (object_cy_rel - roi_cy)**2)
                            if distance <= distance_threshold:
                                trigger_yolo = True

                                if data["display_active"]:
                                    obj_abs_cx = object_cx_rel + effective_x1
                                    obj_abs_cy = object_cy_rel + effective_y1
                                    roi_abs_cx = roi_cx + effective_x1
                                    roi_abs_cy = roi_cy + effective_y1
                                    cv2.rectangle(process_frame, (x_rel + effective_x1, y_rel + effective_y1), (x_rel + w_rel + effective_x1, y_rel + h_rel + effective_y1), (0, 0, 255), 2)
                                    cv2.circle(process_frame, (obj_abs_cx, obj_abs_cy), 5, (0, 255, 255), -1)
                                    cv2.circle(process_frame, (roi_abs_cx, roi_abs_cy), 7, (255, 0, 255), -1)
                                    cv2.line(process_frame, (obj_abs_cx, obj_abs_cy), (roi_abs_cx, roi_abs_cy), (255, 255, 0), 2)

                                break

            data["prev_gray_frame"] = current_gray_frame

        if trigger_yolo and data["frame_count"] % frame_skip == 0:
            log(lane_id, f"🎯 Motion near ROI center detected. Triggering YOLO for lane {lane_id}.")
            try:
                with yolo_lock:
                    conf_threshold = config.get("global", {}).get("yolo_conf", 0.45)
                    results = model.predict(roi_frame_for_yolo, conf=conf_threshold, verbose=False, device="cpu")
                plate_class_id = 0
                boxes = [box for box in results[0].boxes if int(box.cls[0]) == plate_class_id]
            except Exception as e:
                log(lane_id, f"❌ YOLO prediction error after motion in lane {lane_id}: {e}")
                boxes = []

        if boxes:
            data["plate_detected_in_this_frame"] = True
            for box in boxes:
                coords_resized = box.xyxy[0].cpu().numpy()
                x_min_r, y_min_r, x_max_r, y_max_r = map(int, coords_resized)
                x_min_global = x_min_r + effective_x1
                y_min_global = y_min_r + effective_y1
                x_max_global = x_max_r + effective_x1
                y_max_global = y_max_r + effective_y1

                padding = 40
                h, w, _ = current_frame.shape
                x_min_padded = max(0, x_min_global - padding)
                y_min_padded = max(0, y_min_global - padding)
                x_max_padded = min(w - 1, x_max_global + padding)
                y_max_padded = min(h - 1, y_max_global + padding)

                if x_max_padded - x_min_padded > 10 and y_max_padded - y_min_padded > 10:
                    plate_img = current_frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded].copy()
                    if plate_img.size > 0:
                        data["plate_buffer"].append((plate_img, (x_min_global, y_min_global, x_max_global, y_max_global)))
                        if data["display_active"]:
                            cv2.rectangle(process_frame, (x_min_global, y_min_global), (x_max_global, y_max_global), (0, 255, 0), 2)

        if data["plate_detected_in_this_frame"]:
            data["last_vehicle_leaving_time"] = time.time()
        else:
            current_time = time.time()
            if (current_time - data["last_vehicle_leaving_time"]) > vehicle_threshold_time and data["plate_buffer"]:
                try:
                    sharpest_data = max(data["plate_buffer"], key=lambda x: get_sharpness(x[0]))
                    sharpest, coords = sharpest_data
                    sharp_val = get_sharpness(sharpest)
                    if sharp_val > 40:
                        data["ocr_queue"].put((sharpest, coords, sharp_val, current_frame.shape))
                        log(lane_id, f"🎯 Triggered OCR for sharpest plate (sharpness: {sharp_val:.2f}) from buffer in lane {lane_id}.")
                    else:
                        log(lane_id, f"ℹ️ Plate in buffer too blurry for OCR (sharpness: {sharp_val:.2f}) in lane {lane_id}. Skipping.")
                except (ValueError, TypeError):
                    log(lane_id, f"⚠️ Buffer empty or invalid data for sharpest plate in lane {lane_id}.")
                data["plate_buffer"].clear()
            elif not data["plate_buffer"]:
                data["last_vehicle_leaving_time"] = time.time()

        if data["display_active"]:
            cv2.rectangle(process_frame, (effective_x1, effective_y1), (effective_x2, effective_y2), (255, 0, 0), 2)
            display_frame = cv2.resize(process_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            display_queue.put((lane_id, display_frame))

        if data["frame_count"] % 100 == 0 and not data["roi_auto_set"]:
            log(lane_id, f"🔍 Detection progress: {data['detection_counter']}/{heatmap_threshold} for ROI in lane {lane_id}.")

        if data["frame_count"] % 500 == 0:
            gc.collect()
            log_cpu_mem() # Global log

        time.sleep(0.001) # Small sleep to yield CPU

# Start threads for each lane
for lane_id in lane_data:
    if lane_data[lane_id]["use_socket"]:
        threading.Thread(target=connect_to_server, args=(lane_id,)).start()
    threading.Thread(target=ocr_worker, args=(lane_id,), daemon=True).start()
    threading.Thread(target=stream_loop, args=(lane_id,), daemon=True).start()
    threading.Thread(target=processing_loop, args=(lane_id,), daemon=True).start()

# Dictionary to hold the latest display frame for each lane
latest_display_frames = {}
display_frames_lock = threading.Lock()

# Function to update latest frames
def update_latest_frames():
    while not global_shutdown_event.is_set():
        try:
            lane_id, display_frame = display_queue.get(timeout=0.1)
            with display_frames_lock:
                latest_display_frames[lane_id] = display_frame
        except queue.Empty:
            continue

# Start a thread to update latest frames from queue
threading.Thread(target=update_latest_frames, daemon=True).start()

# Function to create combined display frame
def create_combined_frame():
    with display_frames_lock:
        active_lanes = [lid for lid in sorted(latest_display_frames.keys()) if lane_data.get(lid, {}).get("display_active", False)]
        num_active = len(active_lanes)
        if num_active == 0:
            return np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Determine grid: find rows and cols such that rows * cols >= num_active, prefer square
        cols = int(math.ceil(math.sqrt(num_active)))
        rows = int(math.ceil(num_active / cols))

        # Each sub-frame size: divide the main display area equally
        sub_width = DISPLAY_WIDTH // cols
        sub_height = DISPLAY_HEIGHT // rows

        combined = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        for idx, lane_id in enumerate(active_lanes):
            frame = latest_display_frames.get(lane_id)
            if frame is not None:
                # Resize to sub size, preserving aspect ratio by letterboxing if needed
                resized = cv2.resize(frame, (sub_width, sub_height), interpolation=cv2.INTER_AREA)

                row = idx // cols
                col = idx % cols
                y_start = row * sub_height
                x_start = col * sub_width
                combined[y_start:y_start + sub_height, x_start:x_start + sub_width] = resized

                # Add lane label
                cv2.putText(combined, f"Lane {lane_id}", (x_start + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return combined

# Main loop for display and key handling
previous_display_window_active = display_window_active
window_name = "Multi-Lane Detection"
while not global_shutdown_event.is_set():
    try:
        # Check for config file changes
        current_mtime = os.path.getmtime(config_path)
        if current_mtime != config_last_modified:
            reload_config()

        # Handle display window open/close
        if display_window_active != previous_display_window_active:
            if not display_window_active:
                cv2.destroyWindow(window_name)
                global_log("🖼️ Global display window closed due to config change.")
            previous_display_window_active = display_window_active

        # Create and show combined frame only if display is active
        if display_window_active:
            combined_frame = create_combined_frame()
            cv2.imshow(window_name, combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            global_log("🚩 Quitting by 'q' key.")
            global_shutdown_event.set()
            break
        elif ord('0') <= key <= ord('9'):
            pressed_lane = int(chr(key))
            if pressed_lane in lane_data:
                log(pressed_lane, f"🚩 Shutting down lane {pressed_lane} by key '{pressed_lane}'.")
                lane_data[pressed_lane]["shutdown_event"].set()
    except Exception as e:
        global_log(f"❌ Error in main loop: {e}")

# Shutdown all
global_shutdown_event.set()
for lane_id in lane_data:
    lane_data[lane_id]["shutdown_event"].set()
    if lane_data[lane_id]["client_socket"]:
        with lane_data[lane_id]["socket_lock"]:
            try:
                lane_data[lane_id]["client_socket"].shutdown(socket.SHUT_RDWR)
                lane_data[lane_id]["client_socket"].close()
            except OSError as e:
                log(lane_id, f"⚠️ Error during socket shutdown/close for lane {lane_id}: {e}")
cv2.destroyAllWindows()
global_log("✅ All detections stopped. All windows closed.")
