import sys
import math
import time

import os
import json
import cv2
import cv2.aruco as aruco
import numpy as np
from typing import Optional, Tuple

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox

# Thư viện Serial - Cần cài đặt: pip install pyserial
try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("CẢNH BÁO: Không tìm thấy thư viện 'pyserial'. Chức năng Serial sẽ không hoạt động.")
    serial = None

# ============================================================
# IMPORT UI (Giả định file giao_dien_2.py đã tồn tại)
# ============================================================
try:
    from giao_dien_2 import Ui_MainWindow
except ImportError:
    print("LỖI: Không tìm thấy file 'giao_dien_2.py'!")
    raise

# ============================================================
# CẤU HÌNH CAMERA + DETECT
# ============================================================
CAMERA_ID = 3

MARKER_ID = 5
MARKER_SIZE_CM = 48

ROBOT_ORIGIN = np.array([208.0, -115.23, -31.0])
YELLOW_TARGET = np.array([150.0, -285.0, -24.5])
BLUE_TARGET = np.array([70.0, -285.0, -24.5])
RED_TARGET = np.array([-5.0, -285.0, -24.5])
SCALE_X = 1
SCALE_Y = 1
ROI_TOP_Y = 260

COLOR_RANGES = {
    "BLUE": {
        "low": np.array([95, 70, 50]),
        "high": np.array([125, 255, 255]),
        "draw_color": (255, 0, 0)
    },
    "YELLOW": {
        "low": np.array([25, 50, 120]),
        "high": np.array([45, 255, 255]),
        "draw_color": (0, 255, 255)
    },
    "RED": {
        "low1": np.array([0, 90, 70]),   "high1": np.array([10, 255, 255]),
        "low2": np.array([170, 90, 70]), "high2": np.array([180, 255, 255]),
        "draw_color": (0, 0, 255)
    }
}

MIN_AREA = 400
MAX_AREA = 60000

BLUE_BG_DIFF_THRESH = 20
BG_GREEN_LOW  = np.array([75, 40, 40])
BG_GREEN_HIGH = np.array([110, 255, 255])

SHOW_MASKS = False

# ============================================================
# ===== COUNT BOX (Camera 2) - ROI rectangle (KHUNG) =====
# ============================================================
COUNT_BOX_X1_FRAC = 0.2
COUNT_BOX_Y1_FRAC = 0.38
COUNT_BOX_X2_FRAC = 0.5
COUNT_BOX_Y2_FRAC = 0.9
COUNT_COOLDOWN_MS = 500
AUTO_STABILIZE_MS = 1000

# ============================================================
# ARUCO DETECT COMPAT
# ============================================================
def detect_aruco(frame: np.ndarray, dictionary) -> Tuple[list, Optional[np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids

    if hasattr(aruco, "detectMarkers"):
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
        return corners, ids

    raise AttributeError(
        "Aruco API không có detectMarkers/ArucoDetector. "
        "Hãy cài: pip uninstall -y opencv-python opencv-contrib-python && pip install opencv-contrib-python"
    )


def compute_marker_axes(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    c = corners.reshape((4, 2))
    top_left, top_right, _, bottom_left = c

    vec_x = top_right - top_left
    vec_y = bottom_left - top_left

    len_x = np.linalg.norm(vec_x)
    len_y = np.linalg.norm(vec_y)

    if len_x == 0 or len_y == 0:
        return top_left, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0

    unit_x = vec_x / len_x
    unit_y = vec_y / len_y
    avg_len = (len_x + len_y) / 2.0
    return top_left, unit_x, unit_y, avg_len


def build_roi_mask(h: int, w: int) -> np.ndarray:
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[ROI_TOP_Y:, :] = 255
    return roi


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_closest_object(frame: np.ndarray, hsv: np.ndarray, marker_origin: np.ndarray):
    h, w = frame.shape[:2]
    roi_mask = build_roi_mask(h, w)

    bg_mask = cv2.inRange(hsv, BG_GREEN_LOW, BG_GREEN_HIGH)
    bg_inv = cv2.bitwise_not(bg_mask)

    b, g, r = cv2.split(frame)
    diff_bg = cv2.subtract(b, g)
    _, b_dom = cv2.threshold(diff_bg, BLUE_BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)

    best = None
    best_dist = float("inf")

    for color_name, cfg in COLOR_RANGES.items():
        if color_name == "RED":
            m1 = cv2.inRange(hsv, cfg["low1"], cfg["high1"])
            m2 = cv2.inRange(hsv, cfg["low2"], cfg["high2"])
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, cfg["low"], cfg["high"])

        mask = cv2.bitwise_and(mask, roi_mask)

        if color_name == "BLUE":
            mask = cv2.bitwise_and(mask, b_dom)
        else:
            mask = cv2.bitwise_and(mask, bg_inv)

        mask = clean_mask(mask)

        if SHOW_MASKS:
            cv2.imshow(f"mask_{color_name}", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist = float(np.linalg.norm(np.array([cx, cy], dtype=float) - marker_origin))
            if dist < best_dist:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect).astype(np.int32)

                best_dist = dist
                best = {
                    "center": np.array([cx, cy]),
                    "name": color_name,
                    "box": box,
                    "draw_color": cfg["draw_color"],
                    "area": area
                }

    return best


def convert_to_robot_coords(
    obj_px: Tuple[int, int],
    marker_origin: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    pixel_per_cm: float
) -> Tuple[float, float, float]:
    vec = np.array(obj_px, dtype=np.float32) - marker_origin

    rel_x = float(np.dot(vec, uy) / pixel_per_cm)
    rel_y = float(np.dot(vec, ux) / pixel_per_cm)

    final_x = float(ROBOT_ORIGIN[0] + rel_x * SCALE_X)
    final_y = float(ROBOT_ORIGIN[1] + rel_y * SCALE_Y)
    final_z = float(ROBOT_ORIGIN[2])

    return final_x, final_y, final_z


# ============================================================
# OPEN CAMERA ROBUST
# ============================================================
def open_camera(cam_id: int, width=1280, height=720) -> Optional[cv2.VideoCapture]:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_ANY"):
        backends.append(cv2.CAP_ANY)
    backends.append(0)

    for be in backends:
        try:
            cap = cv2.VideoCapture(cam_id, be) if be != 0 else cv2.VideoCapture(cam_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                cap.release()
                continue

            ok = False
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    break

            if ok:
                print(f"[CAM] Open OK: id={cam_id}, backend={be}")
                return cap
            cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            continue

    return None


# ============================================================
# CAMERA WORKER
# ============================================================
class CameraWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    detect_ready = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, cam_id: int, aruco_dict, draw_axes: bool = False, draw_count_line: bool = False, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.aruco_dict = aruco_dict
        self.draw_axes = draw_axes
        self.draw_count_line = draw_count_line
        self._running = False
        self.cap = None
        self._sleep_ms = 50

    def start_cam(self):
        if self._running:
            return
        self._running = True
        self.start()

    def stop_cam(self):
        self._running = False
        self.wait(800)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    def run(self):
        try:
            self.cap = open_camera(self.cam_id, 1280, 720)
            if self.cap is None:
                self.error.emit(f"Không mở được camera {self.cam_id}")
                self._running = False
                return

            while self._running:
                if not self.cap.grab():
                    self.msleep(10)
                    continue

                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    self.msleep(10)
                    continue

                # ===== VẼ COUNT BOX (KHUNG TRẮNG) =====
                count_box = None
                if self.draw_count_line:
                    h0, w0 = frame.shape[:2]

                    x1 = int(w0 * COUNT_BOX_X1_FRAC)
                    y1 = int(h0 * COUNT_BOX_Y1_FRAC)
                    x2 = int(w0 * COUNT_BOX_X2_FRAC)
                    y2 = int(h0 * COUNT_BOX_Y2_FRAC)

                    x1 = max(0, min(w0 - 1, x1))
                    x2 = max(0, min(w0 - 1, x2))
                    y1 = max(0, min(h0 - 1, y1))
                    y2 = max(0, min(h0 - 1, y2))

                    if x2 < x1:
                        x1, x2 = x2, x1
                    if y2 < y1:
                        y1, y2 = y2, y1

                    count_box = (x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.putText(frame, "COUNT BOX", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                try:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # ===== CLAHE kênh V =====
                    h_, s_, v_ = cv2.split(hsv)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    v_ = clahe.apply(v_)
                    hsv = cv2.merge([h_, s_, v_])

                    corners, ids = detect_aruco(frame, self.aruco_dict)

                    marker_origin = None
                    ux = uy = None
                    pixel_per_cm = None

                    if ids is not None:
                        for i, marker_id in enumerate(ids.flatten()):
                            if marker_id == MARKER_ID:
                                c = corners[i][0]
                                marker_origin, ux, uy, px_len = compute_marker_axes(c)
                                pixel_per_cm = px_len / MARKER_SIZE_CM

                                if self.draw_axes:
                                    origin_pt = tuple(marker_origin.astype(int))
                                    axis_x_end = tuple((marker_origin + uy * 100).astype(int))
                                    axis_y_end = tuple((marker_origin + ux * 100).astype(int))
                                    cv2.arrowedLine(frame, origin_pt, axis_x_end, (0, 0, 255), 3, tipLength=0.2)
                                    cv2.arrowedLine(frame, origin_pt, axis_y_end, (0, 255, 0), 3, tipLength=0.2)
                                    cv2.putText(frame, "X_robot", axis_x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(frame, "Y_robot", axis_y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                break

                    closest = None
                    if marker_origin is not None and pixel_per_cm is not None:
                        closest = detect_closest_object(frame, hsv, marker_origin.astype(float))
                        if closest is not None:
                            cx, cy = closest["center"]
                            X, Y, Z = convert_to_robot_coords(
                                (int(cx), int(cy)), marker_origin, ux, uy, pixel_per_cm
                            )
                            closest["robot_xyz"] = (X, Y, Z)

                            # attach count box for cam2 counting
                            if count_box is not None:
                                closest["_count_box"] = count_box

                            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                            cv2.putText(frame, f"{closest['name']} X:{X:.1f} Y:{Y:.1f} Z:{Z:.1f}",
                                        (int(cx) + 10, int(cy) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    self.detect_ready.emit(closest)

                except Exception as e:
                    self.error.emit(f"CameraProc error: {e}")

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
                    self.frame_ready.emit(qimg)
                except Exception:
                    pass

                self.msleep(self._sleep_ms)

        except Exception as e:
            self.error.emit(str(e))
            self._running = False


# ============================================================
# SERIAL RX THREAD
#  FIX: drain buffer -> DONE_2 không bị nằm phía sau log khác
# ============================================================
class ReadSerialThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)

    def __init__(self, serial_port):
        super(ReadSerialThread, self).__init__()
        self.serial_port = serial_port
        self.is_running = True

    def run(self):
        while self.is_running:
            if self.serial_port and self.serial_port.is_open:
                try:
                    # FIX: đọc sạch toàn bộ buffer mỗi vòng
                    while self.serial_port.in_waiting > 0:
                        raw = self.serial_port.readline()
                        data = raw.decode('utf-8', errors='ignore').strip()
                        if data:
                            self.data_received.emit(data)
                except Exception:
                    pass
            self.msleep(1)

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait(800)


# ============================================================
# APP CHÍNH
# ============================================================
class RobotControlApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(RobotControlApp, self).__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        # detect data cam1
        self.detected_x = None
        self.detected_y = None
        self.detected_z = None
        self.detected_color = None

        # detect data cam2
        self.detected_x_2 = None
        self.detected_y_2 = None
        self.detected_z_2 = None
        self.detected_color_2 = None

        # workers
        self.cam_worker_1 = None
        self.cam_worker_2 = None

        # serial
        self.serial_port = None
        self.read_thread = None

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        # ===== vị trí trung gian =====
        self.mid_pose = (219.79, -239.26, -40.5)

        # ===== pbSENDxyz_3 behaviour =====
        self.pb3_auto_mag_enabled = True

        # ===== LOG BUFFER =====
        self._log_buf = []
        self._log_timer = QtCore.QTimer(self)
        self._log_timer.timeout.connect(self._flush_log)
        self._log_timer.start(100)

        # ===== TX QUEUE: gửi 1 lệnh mỗi 30ms =====
        self._tx_queue = []
        self._tx_timer = QtCore.QTimer(self)
        self._tx_timer.timeout.connect(self._tx_pump)
        self._tx_timer.start(30)

        # ===== WAIT TOKEN =====
        self._await_tokens = None
        self._await_deadline_ms = 0
        self._await_tag = ""

        # ===== DELAY SAU DONE =====
        self._post_done_delay_ms = 300
        self._post_done_unlock_ms = 0

        # ===== AUTO MODE / AUTO FLOW =====
        self.auto_enabled = False
        self._auto_busy = False
        self._auto_last_trigger_ms = 0
        self._auto_trigger_cooldown_ms = 800

        # ===== DONE_STOP -> auto click pbSENDxyz_2 sau 1s (anti spam) =====
        self._last_done_stop_ms = 0
        self._done_stop_min_interval_ms = 800

        # ===== COUNT (Cam2) =====
        self.count_yellow = 0
        self.count_blue = 0
        self.count_red = 0

        self._inside_prev = {"YELLOW": False, "BLUE": False, "RED": False}
        self._last_count_ms = {"YELLOW": 0, "BLUE": 0, "RED": 0}

        self._cam1_last_size = None
        self._cam2_last_size = None

        # ===== BUSY flag from UART =====
        self._robot_busy = False
        self._robot_busy_last_ms = 0
        self._robot_busy_timeout_ms = 600

        self._auto_job_color = None
        self._auto_pending_ms = 0
        self._auto_pending_color = None
        self._auto_pending_xyz = None

        # ===== AUTO REFRESH COM PORTS =====
        self._com_port_timer = QtCore.QTimer(self)
        self._com_port_timer.timeout.connect(self._auto_refresh_com_ports)
        self._com_port_timer.start(1000)

        if serial:
            self.setup_serial_ui()

        # CAMERA 1 UI
        if hasattr(self.uic, 'cbCam'):
            self.uic.cbCam.clear()
            self.uic.cbCam.addItems(["0", "1", "2", "3"])
        if hasattr(self.uic, 'pbCC'):
            self.uic.pbCC.clicked.connect(self.ket_noi_camera)
        if hasattr(self.uic, 'pbDC'):
            self.uic.pbDC.clicked.connect(self.ngat_ket_noi_camera)
            self.uic.pbDC.setEnabled(False)

        # CAMERA 2 UI
        if hasattr(self.uic, 'cbCam_2'):
            self.uic.cbCam_2.clear()
            self.uic.cbCam_2.addItems(["0", "1", "2", "3"])
        if hasattr(self.uic, 'pbCC_2'):
            self.uic.pbCC_2.clicked.connect(self.ket_noi_camera_2)
        if hasattr(self.uic, 'pbDC_2'):
            self.uic.pbDC_2.clicked.connect(self.ngat_ket_noi_camera_2)
            self.uic.pbDC_2.setEnabled(False)

        # --- NÚT TÍNH TOÁN ---
        if hasattr(self.uic, 'pbCal'):
            self.uic.pbCal.clicked.connect(self.Tinh_Dong_Hoc_Thuan)
        if hasattr(self.uic, 'pbCalIK'):
            self.uic.pbCalIK.clicked.connect(self.Tinh_Dong_Hoc_Nghich)

        # --- pbCal_2 RESET COUNT ---
        if hasattr(self.uic, 'pbCal_2'):
            self.uic.pbCal_2.clicked.connect(self.reset_count_cam2)

        # --- pbSENDxyz ---
        if hasattr(self.uic, 'pbSENDxyz'):
            self.uic.pbSENDxyz.clicked.connect(self.gui_toa_do_vao_txt_ik)
            self.uic.pbSENDxyz.setEnabled(True)

        # pbSENDxyz_2
        if hasattr(self.uic, 'pbSENDxyz_2'):
            self.uic.pbSENDxyz_2.clicked.connect(self._pbSENDxyz_2_dispatch)
            self.uic.pbSENDxyz_2.setEnabled(True)

        if hasattr(self.uic, 'pbSENDxyz_3'):
            self.uic.pbSENDxyz_3.clicked.connect(self.gui_lenh_xyz_cam2_serial)
            self.uic.pbSENDxyz_3.setEnabled(False)

        # --- NÚT MÀU (manual) ---
        if hasattr(self.uic, 'sendvang'):
            self.uic.sendvang.clicked.connect(lambda: self.start_drop_job("YELLOW"))
        if hasattr(self.uic, 'sendxanh'):
            self.uic.sendxanh.clicked.connect(lambda: self.start_drop_job("BLUE"))
        if hasattr(self.uic, 'senddo'):
            self.uic.senddo.clicked.connect(lambda: self.start_drop_job("RED"))
        if hasattr(self.uic, 'senddo_3'):
            self.uic.senddo_3.clicked.connect(self.gui_lenh_red_target)
        if hasattr(self.uic, 'sendxanh_3'):
            self.uic.sendxanh_3.clicked.connect(self.gui_lenh_blue_target)
        if hasattr(self.uic, 'sendvang_3'):
            self.uic.sendvang_3.clicked.connect(self.gui_lenh_yellow_target)
        if hasattr(self.uic, 'sendvang_7'):
            self.uic.sendvang_7.clicked.connect(self.gui_lenh_robot_origin)
        if hasattr(self.uic, 'sendvang_8'):
            self.uic.sendvang_8.clicked.connect(self.on_sendvang_8_clicked)

        # ===== BUTTON COLOR UI =====
        self._btn_style_default = {"AUTO": ""}
        self._btn_style_active = {
            "AUTO_GRAY": "QPushButton{background:#B0B0B0;color:#222;font-weight:bold;border:2px solid #6E6E6E;border-radius:10px;padding:7px 12px;}",
        }
        self._cache_default_button_styles()
        self._highlight_color_button(None)

        # --- HOME/EMERGENCY/NAMCHAM ---
        if hasattr(self.uic, 'pbHome1'):
            self.uic.pbHome1.clicked.connect(lambda: self.gui_lenh_home("home1"))
            self.uic.pbHome1.setEnabled(False)
        if hasattr(self.uic, 'pbHome1_2'):
            self.uic.pbHome1_2.clicked.connect(lambda: self.gui_lenh_home("home2"))
            self.uic.pbHome1_2.setEnabled(False)
        if hasattr(self.uic, 'pbHome1_3'):
            self.uic.pbHome1_3.clicked.connect(lambda: self.gui_lenh_home("home3"))
            self.uic.pbHome1_3.setEnabled(False)
        if hasattr(self.uic, 'pbHome1_4'):
            self.uic.pbHome1_4.clicked.connect(lambda: self.gui_lenh_home("homeall"))
            self.uic.pbHome1_4.setEnabled(False)
        if hasattr(self.uic, 'pbHome1_5'):
            self.uic.pbHome1_5.clicked.connect(self.gui_lenh_test_voi_spinbox)
            self.uic.pbHome1_5.setEnabled(False)

        if hasattr(self.uic, 'pbD_2'):
            self.uic.pbD_2.clicked.connect(self.emergency_now)
            self.uic.pbD_2.setEnabled(False)
        if hasattr(self.uic, 'pbD_3'):
            self.uic.pbD_3.clicked.connect(self.emergency_now)
            try:
                self.uic.pbD_3.setEnabled(False)
            except Exception:
                pass

        if hasattr(self.uic, 'pbHome1_7'):
            self.uic.pbHome1_7.clicked.connect(self.manual_magnet_on)
            self.uic.pbHome1_7.setEnabled(False)
        if hasattr(self.uic, 'pbHome1_8'):
            self.uic.pbHome1_8.clicked.connect(self.manual_magnet_off)
            self.uic.pbHome1_8.setEnabled(False)

        if hasattr(self.uic, 'pbSendS1'):
            self.uic.pbSendS1.clicked.connect(
                lambda: self.gui_du_lieu_serial_ik(
                    self.uic.txtq1_1, self.uic.txtq2_1, self.uic.txtq3_1, "Bộ 1"
                )
            )
            self.uic.pbSendS1.setEnabled(False)

        if hasattr(self.uic, 'pbSendS2'):
            self.uic.pbSendS2.clicked.connect(
                lambda: self.gui_du_lieu_serial_ik(
                    self.uic.txtq1_2,
                    self.uic.txtq2_1 if hasattr(self.uic, 'txtq2_1') else self.uic.txtq2_2,
                    self.uic.txtq3_2,
                    "Bộ 2"
                )
            )
            self.uic.pbSendS2.setEnabled(False)

        if hasattr(self.uic, 'pbSend'):
            self.uic.pbSend.clicked.connect(
                lambda: self.gui_du_lieu_serial_fk(
                    getattr(self.uic, 'txtq1', None),
                    getattr(self.uic, 'txtq2', None),
                    getattr(self.uic, 'txtq3', None)
                )
            )
            self.uic.pbSend.setEnabled(False)

        if hasattr(self.uic, 'pbman'):
            self.uic.pbman.clicked.connect(self.set_manual_mode)
        if hasattr(self.uic, 'pbauto'):
            self.uic.pbauto.clicked.connect(self.set_auto_mode)

        self._style_btn_on  = "QPushButton{background:#2ECC71;color:#fff;font-weight:bold;border-radius:10px;padding:6px 12px;}"
        self._style_btn_off = "QPushButton{background:#B0B0B0;color:#222;font-weight:bold;border-radius:10px;padding:6px 12px;}"
        self._apply_mode_button_style()

        self._update_count_ui()
        self._load_robot_origin_from_settings()
        self._load_yellow_target_from_settings()
        self._load_blue_target_from_settings()
        self._load_red_target_from_settings()
        self._update_robot_origin_ui()
        self._update_yellow_target_ui()
        self._update_blue_target_ui()
        self._update_red_target_ui()

    # ========================================================
    # EMERGENCY + MANUAL-ONLY MAGNET
    # ========================================================
    def emergency_now(self):
        # FIX: bấm pbD_2/pbD_3 là OUT AUTO ngay lập tức + gửi emergency
        self.auto_enabled = False
        self._auto_busy = False
        self._robot_busy = False
        self._auto_job_color = None
        self._highlight_color_button(None)
        self._apply_mode_button_style()

        self._tx_queue.clear()
        self._await_tokens = None
        self._await_deadline_ms = 0
        self._await_tag = ""
        self._post_done_unlock_ms = 0

        if self.serial_port is None or (not self.serial_port.is_open):
            self._log("EMERGENCY: Serial chưa mở")
            return

        try:
            self.serial_port.write(b"emergency\n")
            self._push_log_html("<span style='color:red;font-weight:bold;'>EMERGENCY: emergency</span>")
        except Exception as e:
            self._log(f"EMERGENCY TX err: {e}")

    def manual_magnet_on(self):
        if self.auto_enabled:
            self._log("MAG1 bị khóa trong AUTO (chỉ MANUAL mới cho bật nam châm)")
            return
        self.gui_lenh_nam_cham(1)

    def manual_magnet_off(self):
        if self.auto_enabled:
            self._log("MAG0 bị khóa trong AUTO (chỉ MANUAL mới cho tắt nam châm)")
            return
        self.gui_lenh_nam_cham(0)

    # ========================================================
    # MODE
    # ========================================================
    def _apply_mode_button_style(self):
        try:
            if hasattr(self.uic, "pbman"):
                self.uic.pbman.setStyleSheet(self._style_btn_on if not self.auto_enabled else self._style_btn_off)
            if hasattr(self.uic, "pbauto"):
                self.uic.pbauto.setStyleSheet(self._style_btn_off if not self.auto_enabled else self._style_btn_on)
        except Exception:
            pass

    def set_manual_mode(self):
        self.auto_enabled = False
        self._auto_busy = False
        self._highlight_color_button(None)
        self._apply_mode_button_style()
        self._log("MODE: MANUAL (pbman)")

    def set_auto_mode(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return
        self.auto_enabled = True
        self._auto_busy = False
        self._apply_mode_button_style()
        self._log("MODE: AUTO (pbauto) - nhận diện màu + tự chạy")

    # ========================================================
    # COUNT UI + RESET
    # ========================================================
    def _update_count_ui(self):
        if hasattr(self.uic, 'txtPx_I_4'):
            self.uic.txtPx_I_4.setText(str(self.count_yellow))
        if hasattr(self.uic, 'txtPy_I_4'):
            self.uic.txtPy_I_4.setText(str(self.count_blue))
        if hasattr(self.uic, 'txtPz_I_4'):
            self.uic.txtPz_I_4.setText(str(self.count_red))

    def reset_count_cam2(self):
        self.count_yellow = 0
        self.count_blue = 0
        self.count_red = 0
        self._inside_prev = {"YELLOW": False, "BLUE": False, "RED": False}
        self._last_count_ms = {"YELLOW": 0, "BLUE": 0, "RED": 0}
        self._update_count_ui()
        self._highlight_color_button(None)
        self._log("RESET COUNT: YELLOW/BLUE/RED = 0")

    # ========================================================
    # LOG BUFFER
    # ========================================================
    def _push_log_html(self, html: str):
        self._log_buf.append(html)
        if len(self._log_buf) > 200:
            self._log_buf = self._log_buf[-120:]

    def _flush_log(self):
        if not self._log_buf:
            return
        try:
            if hasattr(self.uic, 'textBrowser'):
                self.uic.textBrowser.append("<br>".join(self._log_buf))
                self._log_buf.clear()
                cursor = self.uic.textBrowser.textCursor()
                cursor.movePosition(QtGui.QTextCursor.End)
                self.uic.textBrowser.setTextCursor(cursor)
            else:
                self._log_buf.clear()
        except Exception:
            self._log_buf.clear()

    def _log(self, msg: str):
        try:
            if hasattr(self.uic, 'textBrowser'):
                self._push_log_html(f"<span style='color:#777;'>LOG: {msg}</span>")
            else:
                print("LOG:", msg)
        except Exception:
            pass

    # ========================================================
    # TX QUEUE + WAIT TOKEN + delay after DONE
    # ========================================================
    def _tx_pump(self):
        if not (self.serial_port and self.serial_port.is_open):
            self._tx_queue.clear()
            self._await_tokens = None
            self._await_deadline_ms = 0
            self._await_tag = ""
            return

        now = int(time.time() * 1000)

        if now < int(self._post_done_unlock_ms):
            return

        if self._await_tokens is not None:
            if now > self._await_deadline_ms:
                self._log(f"WAIT TIMEOUT ({self._await_tag}) -> bỏ chờ để chạy tiếp")
                self._await_tokens = None
                self._await_deadline_ms = 0
                self._await_tag = ""
            else:
                return

        if not self._tx_queue:
            return

        item = self._tx_queue.pop(0)
        cmd = item["cmd"]
        tag = item.get("tag", "TX")
        wait_tokens = item.get("wait_tokens", None)
        timeout_ms = int(item.get("timeout_ms", 4000))

        try:
            self.serial_port.write(cmd.encode("utf-8"))
            self._push_log_html(f"<span style='color: green;'>{tag}: {cmd.strip()}</span>")

            if wait_tokens:
                if isinstance(wait_tokens, str):
                    wait_tokens = {wait_tokens}
                else:
                    wait_tokens = set(wait_tokens)
                self._await_tokens = wait_tokens
                self._await_deadline_ms = int(time.time() * 1000) + timeout_ms
                self._await_tag = tag

        except Exception as e:
            self._log(f"Serial TX err: {e}")

    def _serial_tx(self, cmd: str, tag: str = "TX", wait_tokens=None, timeout_ms=4000):
        if not (self.serial_port and self.serial_port.is_open):
            self._log("Serial chưa mở")
            return
        if len(self._tx_queue) > 120:
            self._tx_queue.clear()
            self._log("TX queue overflow -> clear")
        self._tx_queue.append({
            "cmd": cmd,
            "tag": tag,
            "wait_tokens": wait_tokens,
            "timeout_ms": timeout_ms
        })

    # ========================================================
    # STEP2 AUTO DISPATCH for pbSENDxyz_2
    # ========================================================
    def _pbSENDxyz_2_dispatch(self):
        if not self.auto_enabled:
            return self.gui_toa_do_vao_txt_ik_2()
        self._auto_try_start_from_pbSENDxyz_2()

    def _is_robot_ready(self) -> bool:
        now = int(time.time() * 1000)
        if self._await_tokens is not None:
            return False
        if self._tx_queue:
            return False
        if self._robot_busy and (now - int(self._robot_busy_last_ms) <= int(self._robot_busy_timeout_ms)):
            return False
        return True

    def _enqueue_drop_sequence_v2(self, job_name: str, speed: int):
        mx, my, mz = self.mid_pose
        self._serial_tx(
            f"L {mx}, {my}, {mz}, {speed}\n",
            tag=f"[AUTO2-{job_name}] MID",
            wait_tokens={"DONE"},
            timeout_ms=3000
        )

        if job_name == "YELLOW":
            tx, ty, tz = (float(YELLOW_TARGET[0]), float(YELLOW_TARGET[1]), float(YELLOW_TARGET[2]))
        elif job_name == "BLUE":
            tx, ty, tz = (float(BLUE_TARGET[0]), float(BLUE_TARGET[1]), float(BLUE_TARGET[2]))
        elif job_name == "RED":
            tx, ty, tz = (float(RED_TARGET[0]), float(RED_TARGET[1]), float(RED_TARGET[2]))
        else:
            tx, ty, tz = (mx, my, mz)

        self._serial_tx(
            f"L {tx}, {ty}, {tz}, {speed}\n",
            tag=f"[AUTO2-{job_name}] TARGET",
            wait_tokens={"DONE"},
            timeout_ms=9000
        )

        self._serial_tx(
            "MAG0\n",
            tag=f"[AUTO2-{job_name}] MAG0",
            wait_tokens={"DONE"},
            timeout_ms=6000
        )

        self._serial_tx(
            "home2\n",
            tag=f"[AUTO2-{job_name}] HOME2",
            wait_tokens={"DONE_2"},
            timeout_ms=8000
        )

    def _auto_try_start_from_pbSENDxyz_2(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return

        color = self.detected_color_2
        if color not in ("YELLOW", "BLUE", "RED"):
            self._highlight_color_button(None)
            return

        if self.detected_x_2 is None or self.detected_y_2 is None or self.detected_z_2 is None:
            return

        self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I_3', None), self.detected_x_2)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I_3', None), self.detected_y_2)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I_3', None), self.detected_z_2)

        self._highlight_color_button(color)

        if not self._is_robot_ready():
            QtCore.QTimer.singleShot(80, self._auto_try_start_from_pbSENDxyz_2)
            return

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        px, py, pz = float(self.detected_x_2), float(self.detected_y_2), float(self.detected_z_2)

        if self._auto_busy:
            return
        self._auto_busy = True
        self._auto_job_color = color

        self._serial_tx(
            f"L {px}, {py}, {pz}, {speed}\n",
            tag="[AUTO2] IK MOVE",
            wait_tokens={"DONE"},
            timeout_ms=14000
        )
        self._serial_tx(
            "MAG1\n",
            tag="[AUTO2] MAG1",
            wait_tokens={"DONE"},
            timeout_ms=10000
        )

        self._enqueue_drop_sequence_v2(color, speed)
        self._serial_tx(";\n", tag="[AUTO2] END_MARK")

    # ========================================================
    # AUTO FLOW (cam2 detect tự chạy)
    # ========================================================
    def _auto_start_pick_and_drop(self, color_name: str, px: float, py: float, pz: float):
        if self._auto_busy:
            return
        if self.serial_port is None or not self.serial_port.is_open:
            return

        now = int(time.time() * 1000)
        if now - int(self._auto_last_trigger_ms) < int(self._auto_trigger_cooldown_ms):
            return
        self._auto_last_trigger_ms = now
        self._auto_busy = True

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        self._serial_tx(
            f"L {px}, {py}, {pz}, {speed}\n",
            tag="[AUTO] PICK MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )
        self._serial_tx(
            "MAG1\n",
            tag="[AUTO] PICK MAG1",
            wait_tokens={"DONE"},
            timeout_ms=5000
        )

        self._enqueue_drop_sequence(color_name, speed)
        self._serial_tx(";\n", tag="[AUTO] END_MARK")

    def _enqueue_drop_sequence(self, job_name: str, speed: int):
        mx, my, mz = self.mid_pose
        self._serial_tx(
            f"L {mx}, {my}, {mz}, {speed}\n",
            tag=f"[AUTO-{job_name}] MID",
            wait_tokens={"DONE"},
            timeout_ms=6000
        )

        if job_name == "YELLOW":
            tx, ty, tz = (float(YELLOW_TARGET[0]), float(YELLOW_TARGET[1]), float(YELLOW_TARGET[2]))
        elif job_name == "BLUE":
            tx, ty, tz = (float(BLUE_TARGET[0]), float(BLUE_TARGET[1]), float(BLUE_TARGET[2]))
        elif job_name == "RED":
            tx, ty, tz = (float(RED_TARGET[0]), float(RED_TARGET[1]), float(RED_TARGET[2]))
        else:
            tx, ty, tz = (mx, my, mz)

        self._serial_tx(
            f"L {tx}, {ty}, {tz}, {speed}\n",
            tag=f"[AUTO-{job_name}] TARGET",
            wait_tokens={"DONE"},
            timeout_ms=6000
        )

        self._serial_tx(
            "MAG0\n",
            tag=f"[AUTO-{job_name}] MAG0",
            wait_tokens={"DONE"},
            timeout_ms=6000
        )

        self._serial_tx(
            "home2\n",
            tag=f"[AUTO-{job_name}] HOME2",
            wait_tokens={"DONE_2"},
            timeout_ms=8000
        )

    def start_drop_job(self, job_name: str):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return
        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()
        self._enqueue_drop_sequence(job_name, speed)

    # ========================================================
    # CAMERA 1
    # ========================================================
    def ket_noi_camera(self):
        cam_id = CAMERA_ID
        if hasattr(self.uic, 'cbCam'):
            try:
                cam_id = int(self.uic.cbCam.currentText())
            except ValueError:
                pass

        if self.cam_worker_1 is not None:
            self.cam_worker_1.stop_cam()
            self.cam_worker_1 = None

        self.cam_worker_1 = CameraWorker(cam_id, self.aruco_dict, draw_axes=True, draw_count_line=False, parent=self)
        self.cam_worker_1.frame_ready.connect(self._on_frame_cam1)
        self.cam_worker_1.detect_ready.connect(self._on_detect_cam1)
        self.cam_worker_1.error.connect(self._log)
        self.cam_worker_1.start_cam()

        if hasattr(self.uic, 'pbCC'):
            self.uic.pbCC.setEnabled(False)
        if hasattr(self.uic, 'pbDC'):
            self.uic.pbDC.setEnabled(True)

    def ngat_ket_noi_camera(self):
        if self.cam_worker_1 is not None:
            self.cam_worker_1.stop_cam()
            self.cam_worker_1 = None

        if hasattr(self.uic, 'lblCam'):
            self.uic.lblCam.clear()
            self.uic.lblCam.setText("Disconnected")
        if hasattr(self.uic, 'pbCC'):
            self.uic.pbCC.setEnabled(True)
        if hasattr(self.uic, 'pbDC'):
            self.uic.pbDC.setEnabled(False)

        self.detected_x = self.detected_y = self.detected_z = None
        self.detected_color = None

    def _on_frame_cam1(self, qimg: QtGui.QImage):
        if not hasattr(self.uic, 'lblCam'):
            return
        size = (self.uic.lblCam.width(), self.uic.lblCam.height())
        self._cam1_last_size = size
        qimg2 = qimg.scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.uic.lblCam.setPixmap(QtGui.QPixmap.fromImage(qimg2))

    def _on_detect_cam1(self, closest):
        if closest is None:
            self.detected_x = self.detected_y = self.detected_z = None
            self.detected_color = None
            return
        self.detected_color = closest.get("name")
        xyz = closest.get("robot_xyz")
        if xyz:
            self.detected_x, self.detected_y, self.detected_z = xyz

    # ========================================================
    # CAMERA 2 + COUNT + AUTO
    # ========================================================
    def ket_noi_camera_2(self):
        cam_id = 0
        if hasattr(self.uic, 'cbCam_2'):
            try:
                cam_id = int(self.uic.cbCam_2.currentText())
            except ValueError:
                pass

        if self.cam_worker_2 is not None:
            self.cam_worker_2.stop_cam()
            self.cam_worker_2 = None

        self.cam_worker_2 = CameraWorker(cam_id, self.aruco_dict, draw_axes=False, draw_count_line=True, parent=self)
        self.cam_worker_2.frame_ready.connect(self._on_frame_cam2)
        self.cam_worker_2.detect_ready.connect(self._on_detect_cam2)
        self.cam_worker_2.error.connect(self._log)
        self.cam_worker_2.start_cam()

        if hasattr(self.uic, 'pbCC_2'):
            self.uic.pbCC_2.setEnabled(False)
        if hasattr(self.uic, 'pbDC_2'):
            self.uic.pbDC_2.setEnabled(True)

    def ngat_ket_noi_camera_2(self):
        if self.cam_worker_2 is not None:
            self.cam_worker_2.stop_cam()
            self.cam_worker_2 = None

        if hasattr(self.uic, 'lblCam_2'):
            self.uic.lblCam_2.clear()
            self.uic.lblCam_2.setText("Disconnected")
        if hasattr(self.uic, 'pbCC_2'):
            self.uic.pbCC_2.setEnabled(True)
        if hasattr(self.uic, 'pbDC_2'):
            self.uic.pbDC_2.setEnabled(False)

        self.detected_x_2 = self.detected_y_2 = self.detected_z_2 = None
        self.detected_color_2 = None
        self._inside_prev = {"YELLOW": False, "BLUE": False, "RED": False}
        self._highlight_color_button(None)

    def _on_frame_cam2(self, qimg: QtGui.QImage):
        if not hasattr(self.uic, 'lblCam_2'):
            return
        size = (self.uic.lblCam_2.width(), self.uic.lblCam_2.height())
        self._cam2_last_size = size
        qimg2 = qimg.scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.uic.lblCam_2.setPixmap(QtGui.QPixmap.fromImage(qimg2))

    def _on_detect_cam2(self, closest):
        if closest is None:
            self.detected_x_2 = self.detected_y_2 = self.detected_z_2 = None
            self.detected_color_2 = None
            self._auto_pending_ms = 0
            self._auto_pending_color = None
            self._auto_pending_xyz = None
            return

        self.detected_color_2 = closest.get("name")
        xyz = closest.get("robot_xyz")
        if xyz:
            self.detected_x_2, self.detected_y_2, self.detected_z_2 = xyz

        # COUNT BOX
        inside_box = False
        name = closest.get("name")
        center = closest.get("center")
        box = closest.get("_count_box", None)

        if name in ("YELLOW", "BLUE", "RED") and center is not None and box is not None:
            cx, cy = int(center[0]), int(center[1])
            x1, y1, x2, y2 = box

            inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            inside_box = inside
            prev_inside = bool(self._inside_prev.get(name, False))
            self._inside_prev[name] = inside

            if (not prev_inside) and inside:
                now = int(time.time() * 1000)
                if now - int(self._last_count_ms[name]) >= int(COUNT_COOLDOWN_MS):
                    self._last_count_ms[name] = now
                    if name == "YELLOW":
                        self.count_yellow += 1
                    elif name == "BLUE":
                        self.count_blue += 1
                    elif name == "RED":
                        self.count_red += 1
                    self._update_count_ui()
                    self._highlight_color_button(name)

        if self.auto_enabled and not inside_box:
            self.detected_x_2 = self.detected_y_2 = self.detected_z_2 = None
            self.detected_color_2 = None
            self._auto_pending_ms = 0
            self._auto_pending_color = None
            self._auto_pending_xyz = None
            return

        # AUTO
        if not self.auto_enabled:
            return
        if self._auto_busy:
            return
        if self.serial_port is None or not self.serial_port.is_open:
            return
        if name not in ("YELLOW", "BLUE", "RED"):
            return
        if not inside_box:
            return
        if xyz is None:
            return

        now = int(time.time() * 1000)
        if self._auto_pending_color != name or not self._auto_pending_ms:
            self._auto_pending_color = name
            self._auto_pending_ms = now
            self._auto_pending_xyz = xyz
            return
        self._auto_pending_xyz = xyz
        if now - int(self._auto_pending_ms) < int(AUTO_STABILIZE_MS):
            px, py, pz = self._auto_pending_xyz
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I_3', None), px)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I_3', None), py)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I_3', None), pz)
            return

        px, py, pz = self._auto_pending_xyz
        self._auto_pending_ms = 0
        self._auto_pending_color = None
        self._auto_pending_xyz = None
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I_3', None), px)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I_3', None), py)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I_3', None), pz)

        if not self._is_robot_ready():
            return

        self._auto_start_pick_and_drop(name, px, py, pz)
        self._log(f"AUTO: detect {name} -> PICK + DROP")

    # ========================================================
    # SERIAL UI
    # ========================================================
    def cap_nhat_com_ports(self):
        if not serial or not hasattr(self.uic, 'comboBox'):
            return
        self.uic.comboBox.clear()
        ports = serial.tools.list_ports.comports()
        com_list = [p.device for p in ports]
        if com_list:
            self.uic.comboBox.addItems(com_list)
        else:
            self.uic.comboBox.addItem("Không tìm thấy cổng nào")

    def _auto_refresh_com_ports(self):
        if not serial or not hasattr(self.uic, 'comboBox'):
            return
        if self.serial_port is not None and self.serial_port.is_open:
            return

        ports = serial.tools.list_ports.comports()
        com_list = [p.device for p in ports]

        current_items = [self.uic.comboBox.itemText(i) for i in range(self.uic.comboBox.count())]
        if current_items == com_list:
            return

        current_sel = self.uic.comboBox.currentText()
        self.uic.comboBox.clear()
        if com_list:
            self.uic.comboBox.addItems(com_list)
            if current_sel in com_list:
                self.uic.comboBox.setCurrentText(current_sel)
            elif len(com_list) == 1:
                self.uic.comboBox.setCurrentText(com_list[0])
        else:
            self.uic.comboBox.addItem("Không tìm thấy cổng nào")

    def setup_serial_ui(self):
        self.cap_nhat_com_ports()
        if hasattr(self.uic, 'comboBox_3'):
            self.uic.comboBox_3.addItems(["115200", "9600"])
            self.uic.comboBox_3.setCurrentText("115200")
        if hasattr(self.uic, 'pushButton'):
            self.uic.pushButton.setText("Connect")
            self.uic.pushButton.clicked.connect(self.toggle_serial_connection)
        if hasattr(self.uic, 'pbDisconnectSerial'):
            self.uic.pbDisconnectSerial.clicked.connect(self.ngat_ket_noi_serial)
            self.uic.pbDisconnectSerial.setEnabled(False)

    def toggle_serial_connection(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.ket_noi_serial()
        else:
            self.ngat_ket_noi_serial()

    def ket_noi_serial(self):
        if not serial:
            return

        port_name = self.lay_du_lieu(getattr(self.uic, 'comboBox', None))
        baud_rate_str = self.lay_du_lieu(getattr(self.uic, 'comboBox_3', None))

        if not port_name or not baud_rate_str or "Không tìm thấy" in port_name:
            self.hien_thi_loi("Vui lòng chọn cổng Port và Baudrate.")
            return

        try:
            baud_rate = int(baud_rate_str)
            self.serial_port = serial.Serial(port_name, baud_rate, timeout=0.1, write_timeout=0.1)

            self.read_thread = ReadSerialThread(self.serial_port)
            self.read_thread.data_received.connect(self.xu_ly_du_lieu_serial_rx)
            self.read_thread.start()

            self.hien_thi_thong_bao("Serial", f"Đã kết nối Serial tại {port_name}")

            if hasattr(self.uic, 'pushButton'):
                self.uic.pushButton.setText("Disconnect")

            btn_list = [
                'pbSend', 'pbSendS1', 'pbSendS2', 'pbHome1', 'pbHome1_2',
                'pbHome1_3', 'pbHome1_4', 'pbHome1_5',
                'pbD_2', 'pbD_3',
                'pbHome1_7', 'pbHome1_8',
                'pbSENDxyz_3', 'sendvang', 'sendxanh', 'senddo',
                'pbSENDxyz_2'
            ]
            for btn_name in btn_list:
                if hasattr(self.uic, btn_name):
                    getattr(self.uic, btn_name).setEnabled(True)

            if hasattr(self.uic, 'comboBox'):
                self.uic.comboBox.setEnabled(False)
            if hasattr(self.uic, 'comboBox_3'):
                self.uic.comboBox_3.setEnabled(False)

        except Exception as e:
            self.hien_thi_loi(f"Lỗi kết nối Serial: {e}")

    def ngat_ket_noi_serial(self):
        if self.read_thread is not None:
            self.read_thread.stop()
            self.read_thread = None

        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
        except Exception:
            pass
        self.serial_port = None

        self._tx_queue.clear()
        self._await_tokens = None
        self._await_deadline_ms = 0
        self._await_tag = ""
        self._post_done_unlock_ms = 0
        self.auto_enabled = False
        self._auto_busy = False
        self._robot_busy = False

        self.hien_thi_thong_bao("Serial", "Đã ngắt kết nối Serial.")

        if hasattr(self.uic, 'pushButton'):
            self.uic.pushButton.setText("Connect")

        btn_list = [
            'pbSend', 'pbSendS1', 'pbSendS2', 'pbHome1', 'pbHome1_2',
            'pbHome1_3', 'pbHome1_4', 'pbHome1_5',
            'pbD_2', 'pbD_3',
            'pbHome1_7', 'pbHome1_8',
            'pbSENDxyz_3', 'sendvang', 'sendxanh', 'senddo',
            'pbSENDxyz_2'
        ]
        for btn_name in btn_list:
            if hasattr(self.uic, btn_name):
                getattr(self.uic, btn_name).setEnabled(False)

        if hasattr(self.uic, 'comboBox'):
            self.uic.comboBox.setEnabled(True)
        if hasattr(self.uic, 'comboBox_3'):
            self.uic.comboBox_3.setEnabled(True)
        self.cap_nhat_com_ports()

        self._apply_mode_button_style()
        self._highlight_color_button(None)

    # ========================================================
    # HOME/NAMCHAM/SEND XYZ
    # ========================================================
    def gui_lenh_home(self, command_str):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return
        cmd = (command_str or "").strip().lower()
        self._serial_tx(f"{cmd}\n", tag="HOME")

    def gui_lenh_nam_cham(self, state):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return
        state = 1 if int(state) != 0 else 0
        self._serial_tx(f"MAG{state}\n", tag=f"MAG{state}", wait_tokens={"DONE"}, timeout_ms=4000)

    def gui_lenh_xyz_cam2_serial(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial!")
            return

        px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I_2', None))
        py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I_2', None))
        pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I_2', None))

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        if not px or not py or not pz:
            self.hien_thi_loi("Thiếu dữ liệu tọa độ (Px, Py, Pz) của Camera 2!")
            return

        self._serial_tx(
            f"L {px}, {py}, {pz}, {speed}\n",
            tag="PB_SENDXYZ_3 MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )

        if self.pb3_auto_mag_enabled:
            self._serial_tx(
                "MAG1\n",
                tag="PB_SENDXYZ_3 MAG1",
                wait_tokens={"DONE"},
                timeout_ms=5000
            )

    def gui_lenh_test_voi_spinbox(self):
        speed = 10
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()
        self.gui_lenh_home(f"test,{speed}")

    # ========================================================
    # SERIAL RX
    #  FIX: time ms đúng + DONE_2 xử lý nhanh
    # ========================================================
    def xu_ly_du_lieu_serial_rx(self, data):
        if hasattr(self.uic, 'textBrowser'):
            self._push_log_html(f"<span style='font-size: 10px; color: #2E86C1;'>RX: {data}</span>")

        d = data.strip()

        if d.lower().startswith("busy"):
            self._robot_busy = True
            self._robot_busy_last_ms = int(time.time() * 1000)  # FIX

        if d == "DONE_STOP":
            if not self.auto_enabled:
                now = int(time.time() * 1000)
                if now - int(self._last_done_stop_ms) >= int(self._done_stop_min_interval_ms):
                    self._last_done_stop_ms = now
                    QtCore.QTimer.singleShot(1000, lambda: (
                        self.uic.pbSENDxyz_2.click()
                        if hasattr(self.uic, "pbSENDxyz_2") and self.uic.pbSENDxyz_2.isEnabled()
                        else None
                    ))

        # WAIT TOKEN
        if self._await_tokens is not None and d in self._await_tokens:
            token = d
            self._log(f"WAIT OK ({self._await_tag}) <- {token}")
            self._await_tokens = None
            self._await_deadline_ms = 0
            self._await_tag = ""

            now = int(time.time() * 1000)  # FIX

            # chỉ delay cho DONE, không delay DONE_2
            if token == "DONE":
                self._post_done_unlock_ms = now + int(self._post_done_delay_ms)
            else:
                self._post_done_unlock_ms = now

        # DONE_2: kết thúc hoàn toàn -> KHÔNG retry (đứng yên chờ detect mới chạy)
        if d == "DONE_2":
            self._robot_busy = False
            if self.auto_enabled:
                self._auto_busy = False
                self._highlight_color_button(None)
                # FIX: bỏ auto re-try để không gửi thêm IK khi chưa có vật trong count box
                # QtCore.QTimer.singleShot(20, self._auto_try_start_from_pbSENDxyz_2)

        if d.startswith("MAG_STATUS:"):
            try:
                status = int(d.split(':')[1].strip())
                if hasattr(self.uic, 'lblMagnetStatus'):
                    text = "BẬT (Có vật)" if status == 1 else "TẮT (Không vật)"
                    color = "green" if status == 1 else "red"
                    self.uic.lblMagnetStatus.setText(text)
                    self.uic.lblMagnetStatus.setStyleSheet(f"color: {color}; font-weight: bold;")
            except Exception:
                pass

    # ========================================================
    # SEND IK/FK
    # ========================================================
    def gui_du_lieu_serial_ik(self, widget_q1, widget_q2, widget_q3, bo_ket_qua):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial.")
            return
        try:
            s_q1 = self.lay_du_lieu(widget_q1)
            s_q2 = self.lay_du_lieu(widget_q2)
            s_q3 = self.lay_du_lieu(widget_q3)

            speed = 50
            if hasattr(self.uic, 'spinBox'):
                speed = self.uic.spinBox.value()

            if not s_q1 or not s_q2 or not s_q3:
                self.hien_thi_loi(f"Thiếu dữ liệu {bo_ket_qua}!")
                return

            q1 = float(s_q1)
            q2 = float(s_q2)
            q3 = float(s_q3)

            data_to_send = f"{q1:.2f},{q2:.2f},{q3:.2f},{speed}\n"
            self._serial_tx(data_to_send, tag="IK")

        except ValueError:
            self.hien_thi_loi("Dữ liệu phải là số!")
        except Exception as e:
            self.hien_thi_loi(f"Lỗi gửi Serial: {e}")

    def gui_du_lieu_serial_fk(self, widget_q1, widget_q2, widget_q3):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chưa kết nối Serial.")
            return
        try:
            s_q1 = self.lay_du_lieu(widget_q1)
            s_q2 = self.lay_du_lieu(widget_q2)
            s_q3 = self.lay_du_lieu(widget_q3)

            speed = 50
            if hasattr(self.uic, 'spinBox'):
                speed = self.uic.spinBox.value()

            if not s_q1 or not s_q2 or not s_q3:
                self.hien_thi_loi("Thiếu dữ liệu FK!")
                return

            q1 = float(s_q1)
            q2 = float(s_q2)
            q3 = float(s_q3)

            data_to_send = f"{q1:.2f},{q2:.2f},{q3:.2f},{speed}\n"
            self._serial_tx(data_to_send, tag="FK")

        except ValueError:
            self.hien_thi_loi("Dữ liệu phải là số!")
        except Exception as e:
            self.hien_thi_loi(f"Lỗi gửi Serial: {e}")

    # ========================================================
    # pbSENDxyz / pbSENDxyz_2
    # ========================================================
    def gui_toa_do_vao_txt_ik(self):
        if self.detected_x is None or self.detected_y is None or self.detected_z is None:
            self.hien_thi_loi("Chưa phát hiện vật hoặc tọa độ chưa sẵn sàng.")
            return
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I', None), self.detected_x)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I', None), self.detected_y)
        self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I', None), self.detected_z)

    def gui_toa_do_vao_txt_ik_2(self):
        if self.detected_x_2 is None or self.detected_y_2 is None or self.detected_z_2 is None:
            self.hien_thi_loi("Chưa phát hiện vật (Cam 2) hoặc tọa độ chưa sẵn sàng.")
            return

        if self.auto_enabled:
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I_3', None), self.detected_x_2)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I_3', None), self.detected_y_2)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I_3', None), self.detected_z_2)
        else:
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPx_I_2', None), self.detected_x_2)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPy_I_2', None), self.detected_y_2)
            self.hien_thi_ket_qua(getattr(self.uic, 'txtPz_I_2', None), self.detected_z_2)

    # ========================================================
    # UI helpers + highlight
    # ========================================================
    def _cache_default_button_styles(self):
        try:
            if hasattr(self.uic, "sendauto"):
                self._btn_style_default["AUTO"] = self.uic.sendauto.styleSheet() or ""
        except Exception:
            pass

    def _highlight_color_button(self, color_name: str):
        try:
            if hasattr(self.uic, "sendauto"):
                self.uic.sendauto.setStyleSheet(self._btn_style_active["AUTO_GRAY"])

            if color_name in ("YELLOW", "BLUE", "RED") and hasattr(self.uic, "sendauto"):
                if color_name == "YELLOW":
                    self.uic.sendauto.setStyleSheet(
                        "QPushButton{background:#FFD400;color:#7A5A00;font-weight:bold;border:2px solid #B38F00;border-radius:10px;padding:7px 12px;}"
                    )
                elif color_name == "BLUE":
                    self.uic.sendauto.setStyleSheet(
                        "QPushButton{background:#0047AB;color:#EAF2FF;font-weight:bold;border:2px solid #002D73;border-radius:10px;padding:7px 12px;}"
                    )
                elif color_name == "RED":
                    self.uic.sendauto.setStyleSheet(
                        "QPushButton{background:#E53935;color:#FFECEC;font-weight:bold;border:2px solid #A61B1B;border-radius:10px;padding:7px 12px;}"
                    )
        except Exception:
            pass

    def lay_du_lieu(self, widget):
        if widget is None:
            return ""
        # QComboBox phải lấy currentText() trước, nếu không sẽ đọc sai dữ liệu port/baudrate
        if hasattr(widget, 'currentText'):
            return widget.currentText()
        elif hasattr(widget, 'toPlainText'):
            return widget.toPlainText()
        elif hasattr(widget, 'text'):
            return widget.text()
        else:
            return ""

    def hien_thi_ket_qua(self, widget, gia_tri):
        if widget is None:
            return
        try:
            val = str(round(float(gia_tri), 2))
        except Exception:
            val = str(gia_tri)
        if hasattr(widget, 'setText'):
            widget.setText(val)
        elif hasattr(widget, 'setPlainText'):
            widget.setPlainText(val)

    def safe_sqrt(self, x):
        return math.sqrt(max(0, x))

    def _update_robot_origin_ui(self):
        if hasattr(self.uic, 'txtPx_I_17'):
            self.hien_thi_ket_qua(self.uic.txtPx_I_17, ROBOT_ORIGIN[0])
        if hasattr(self.uic, 'txtPy_I_17'):
            self.hien_thi_ket_qua(self.uic.txtPy_I_17, ROBOT_ORIGIN[1])
        if hasattr(self.uic, 'txtPz_I_17'):
            self.hien_thi_ket_qua(self.uic.txtPz_I_17, ROBOT_ORIGIN[2])

    def _update_red_target_ui(self):
        if hasattr(self.uic, 'txtPx_I_16'):
            self.hien_thi_ket_qua(self.uic.txtPx_I_16, RED_TARGET[0])
        if hasattr(self.uic, 'txtPy_I_16'):
            self.hien_thi_ket_qua(self.uic.txtPy_I_16, RED_TARGET[1])
        if hasattr(self.uic, 'txtPz_I_16'):
            self.hien_thi_ket_qua(self.uic.txtPz_I_16, RED_TARGET[2])

    def _update_blue_target_ui(self):
        if hasattr(self.uic, 'txtPx_I_7'):
            self.hien_thi_ket_qua(self.uic.txtPx_I_7, BLUE_TARGET[0])
        if hasattr(self.uic, 'txtPy_I_7'):
            self.hien_thi_ket_qua(self.uic.txtPy_I_7, BLUE_TARGET[1])
        if hasattr(self.uic, 'txtPz_I_7'):
            self.hien_thi_ket_qua(self.uic.txtPz_I_7, BLUE_TARGET[2])

    def _update_yellow_target_ui(self):
        if hasattr(self.uic, 'txtPx_I_6'):
            self.hien_thi_ket_qua(self.uic.txtPx_I_6, YELLOW_TARGET[0])
        if hasattr(self.uic, 'txtPy_I_6'):
            self.hien_thi_ket_qua(self.uic.txtPy_I_6, YELLOW_TARGET[1])
        if hasattr(self.uic, 'txtPz_I_6'):
            self.hien_thi_ket_qua(self.uic.txtPz_I_6, YELLOW_TARGET[2])

    def _parse_xyz_from_widgets(self, w_px, w_py, w_pz, label):
        px = self.lay_du_lieu(w_px)
        py = self.lay_du_lieu(w_py)
        pz = self.lay_du_lieu(w_pz)

        if not px or not py or not pz:
            self.hien_thi_loi(f"Thieu du lieu {label} (Px, Py, Pz).")
            return None

        try:
            return float(px), float(py), float(pz)
        except ValueError:
            self.hien_thi_loi(f"Du lieu {label} phai la so.")
            return None

    def apply_origin_and_red_target_from_ui(self):
        origin = self._parse_xyz_from_widgets(
            getattr(self.uic, 'txtPx_I_17', None),
            getattr(self.uic, 'txtPy_I_17', None),
            getattr(self.uic, 'txtPz_I_17', None),
            "GOC"
        )
        if origin is None:
            return

        yellow_target = self._parse_xyz_from_widgets(
            getattr(self.uic, 'txtPx_I_6', None),
            getattr(self.uic, 'txtPy_I_6', None),
            getattr(self.uic, 'txtPz_I_6', None),
            "VANG"
        )
        if yellow_target is None:
            return

        blue_target = self._parse_xyz_from_widgets(
            getattr(self.uic, 'txtPx_I_7', None),
            getattr(self.uic, 'txtPy_I_7', None),
            getattr(self.uic, 'txtPz_I_7', None),
            "XANH"
        )
        if blue_target is None:
            return

        red_target = self._parse_xyz_from_widgets(
            getattr(self.uic, 'txtPx_I_16', None),
            getattr(self.uic, 'txtPy_I_16', None),
            getattr(self.uic, 'txtPz_I_16', None),
            "DO"
        )
        if red_target is None:
            return

        ROBOT_ORIGIN[0], ROBOT_ORIGIN[1], ROBOT_ORIGIN[2] = origin
        YELLOW_TARGET[0], YELLOW_TARGET[1], YELLOW_TARGET[2] = yellow_target
        BLUE_TARGET[0], BLUE_TARGET[1], BLUE_TARGET[2] = blue_target
        RED_TARGET[0], RED_TARGET[1], RED_TARGET[2] = red_target
        self._update_robot_origin_ui()
        self._update_yellow_target_ui()
        self._update_blue_target_ui()
        self._update_red_target_ui()

        data = self._load_settings()
        data["robot_origin"] = list(origin)
        data["yellow_target"] = list(yellow_target)
        data["blue_target"] = list(blue_target)
        data["red_target"] = list(red_target)
        if self._save_settings(data):
            self._log(f"SET ORIGIN+TARGETS: {origin} / {yellow_target} / {blue_target} / {red_target}")
        else:
            self._log(f"SET ORIGIN+TARGETS (no save): {origin} / {yellow_target} / {blue_target} / {red_target}")

    def on_sendvang_8_clicked(self):
        # Exit AUTO and go homeall before applying new targets
        if self.auto_enabled:
            self.auto_enabled = False
            self._auto_busy = False
            self._robot_busy = False
            self._auto_job_color = None
            self._highlight_color_button(None)
            self._apply_mode_button_style()
        if self.serial_port is not None and self.serial_port.is_open:
            self.emergency_now()
            self.gui_lenh_home("homeall")
        self.apply_origin_and_red_target_from_ui()

    def _settings_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "settings.json")

    def _load_settings(self):
        try:
            settings_path = self._settings_path()
            if not os.path.exists(settings_path):
                return {}
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _save_settings(self, data):
        try:
            settings_path = self._settings_path()
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            self.hien_thi_loi(f"Khong ghi duoc settings.json: {e}")
            return False

    def _load_robot_origin_from_settings(self):
        try:
            data = self._load_settings()
            origin = data.get("robot_origin")
            if not origin or len(origin) != 3:
                return
            ROBOT_ORIGIN[0] = float(origin[0])
            ROBOT_ORIGIN[1] = float(origin[1])
            ROBOT_ORIGIN[2] = float(origin[2])
        except Exception:
            return

    def _load_red_target_from_settings(self):
        try:
            data = self._load_settings()
            target = data.get("red_target")
            if not target or len(target) != 3:
                return
            RED_TARGET[0] = float(target[0])
            RED_TARGET[1] = float(target[1])
            RED_TARGET[2] = float(target[2])
        except Exception:
            return

    def _load_blue_target_from_settings(self):
        try:
            data = self._load_settings()
            target = data.get("blue_target")
            if not target or len(target) != 3:
                return
            BLUE_TARGET[0] = float(target[0])
            BLUE_TARGET[1] = float(target[1])
            BLUE_TARGET[2] = float(target[2])
        except Exception:
            return

    def _load_yellow_target_from_settings(self):
        try:
            data = self._load_settings()
            target = data.get("yellow_target")
            if not target or len(target) != 3:
                return
            YELLOW_TARGET[0] = float(target[0])
            YELLOW_TARGET[1] = float(target[1])
            YELLOW_TARGET[2] = float(target[2])
        except Exception:
            return

    def gui_lenh_blue_target(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chua ket noi Serial!")
            return

        px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I_7', None)) or str(BLUE_TARGET[0])
        py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I_7', None)) or str(BLUE_TARGET[1])
        pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I_7', None)) or str(BLUE_TARGET[2])

        try:
            px_f = float(px)
            py_f = float(py)
            pz_f = float(pz)
        except ValueError:
            self.hien_thi_loi("Du lieu XANH phai la so.")
            return

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        self._serial_tx(
            f"L {px_f}, {py_f}, {pz_f}, {speed}\n",
            tag="BLUE TARGET MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )

    def gui_lenh_yellow_target(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chua ket noi Serial!")
            return

        px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I_6', None)) or str(YELLOW_TARGET[0])
        py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I_6', None)) or str(YELLOW_TARGET[1])
        pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I_6', None)) or str(YELLOW_TARGET[2])

        try:
            px_f = float(px)
            py_f = float(py)
            pz_f = float(pz)
        except ValueError:
            self.hien_thi_loi("Du lieu VANG phai la so.")
            return

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        self._serial_tx(
            f"L {px_f}, {py_f}, {pz_f}, {speed}\n",
            tag="YELLOW TARGET MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )

    def gui_lenh_red_target(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chua ket noi Serial!")
            return

        px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I_16', None)) or str(RED_TARGET[0])
        py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I_16', None)) or str(RED_TARGET[1])
        pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I_16', None)) or str(RED_TARGET[2])

        try:
            px_f = float(px)
            py_f = float(py)
            pz_f = float(pz)
        except ValueError:
            self.hien_thi_loi("Du lieu DO phai la so.")
            return

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        self._serial_tx(
            f"L {px_f}, {py_f}, {pz_f}, {speed}\n",
            tag="RED TARGET MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )

    def gui_lenh_robot_origin(self):
        if self.serial_port is None or not self.serial_port.is_open:
            self.hien_thi_loi("Chua ket noi Serial!")
            return

        px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I_17', None)) or str(ROBOT_ORIGIN[0])
        py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I_17', None)) or str(ROBOT_ORIGIN[1])
        pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I_17', None)) or str(ROBOT_ORIGIN[2])

        try:
            px_f = float(px)
            py_f = float(py)
            pz_f = float(pz)
        except ValueError:
            self.hien_thi_loi("Du lieu GOC phai la so.")
            return

        speed = 50
        if hasattr(self.uic, 'spinBox'):
            speed = self.uic.spinBox.value()

        self._serial_tx(
            f"L {px_f}, {py_f}, {pz_f}, {speed}\n",
            tag="ORIGIN MOVE",
            wait_tokens={"DONE"},
            timeout_ms=12000
        )

    def hien_thi_loi(self, msg):
        QMessageBox.warning(self, "Lỗi", msg)

    def hien_thi_thong_bao(self, title, msg):
        QMessageBox.information(self, title, msg)

    # ========================================================
    # FK / IK
    # ========================================================
    def Tinh_Dong_Hoc_Thuan(self):
        try:
            if hasattr(self.uic, 'txtq3'):
                s_q3 = self.lay_du_lieu(self.uic.txtq3)
            elif hasattr(self.uic, 'txt3'):
                s_q3 = self.lay_du_lieu(self.uic.txt3)
            else:
                self.hien_thi_loi("Lỗi UI: Không tìm thấy ô nhập góc 3")
                return

            s_q1 = self.lay_du_lieu(self.uic.txtq1)
            s_q2 = self.lay_du_lieu(self.uic.txtq2)
            if not s_q1 or not s_q2 or not s_q3:
                self.hien_thi_loi("Thiếu dữ liệu FK!")
                return

            the1 = math.radians(float(s_q1))
            the3 = math.radians(float(s_q3))
            d2 = float(s_q2)

            l1 = 45
            l2 = 228
            l3 = 63.5
            l4 = 144
            l5 = 30

            Px = l4 * math.cos(the1 + the3) + l2 * math.cos(the1)
            Py = l4 * math.sin(the1 + the3) + l2 * math.sin(the1)
            Pz = d2 + l1 - l3 - l5

            self.hien_thi_ket_qua(self.uic.txtPx, Px)
            self.hien_thi_ket_qua(self.uic.txtPy, Py)
            self.hien_thi_ket_qua(self.uic.txtPz, Pz)
        except ValueError:
            self.hien_thi_loi("Dữ liệu phải là số!")

    def Tinh_Dong_Hoc_Nghich(self):
        try:
            s_px = self.lay_du_lieu(getattr(self.uic, 'txtPx_I', None))
            s_py = self.lay_du_lieu(getattr(self.uic, 'txtPy_I', None))
            s_pz = self.lay_du_lieu(getattr(self.uic, 'txtPz_I', None))

            if not s_px or not s_py or not s_pz:
                self.hien_thi_loi("Thiếu dữ liệu IK!")
                return

            Px = float(s_px)
            Py = float(s_py)
            Pz = float(s_pz)

            l1 = 45
            l2 = 228
            l3 = 63.5
            l4 = 144
            l5 = 30

            d2 = Pz - (l1 - l3 - l5)
            c3 = (Px**2 + Py**2 - l4**2 - l2**2) / (2 * l2 * l4)
            if abs(c3) > 1.0001:
                self.hien_thi_loi("Ngoài vùng làm việc!")
                return
            c3 = max(-1, min(1, c3))

            s3 = self.safe_sqrt(1 - c3**2)
            the3_1 = math.atan2(s3, c3)
            the3_2 = math.atan2(-s3, c3)

            def get_the1(Px_, Py_, the3_):
                val_a = l2 + l4 * math.cos(the3_)
                val_b = l4 * math.sin(the3_)
                return math.atan2(Py_, Px_) - math.atan2(val_b, val_a)

            sol1 = [get_the1(Px, Py, the3_1), d2, the3_1]
            sol2 = [get_the1(Px, Py, the3_2), d2, the3_2]

            self.hien_thi_ket_qua(getattr(self.uic, 'txtq1_1', None), math.degrees(sol1[0]))
            self.hien_thi_ket_qua(getattr(self.uic, 'txtq2_1', None), sol1[1])
            self.hien_thi_ket_qua(getattr(self.uic, 'txtq3_1', None), math.degrees(sol1[2]))

            self.hien_thi_ket_qua(getattr(self.uic, 'txtq1_2', None), math.degrees(sol2[0]))
            self.hien_thi_ket_qua(getattr(self.uic, 'txtq2_2', None), sol2[1])
            self.hien_thi_ket_qua(getattr(self.uic, 'txtq3_2', None), math.degrees(sol2[2]))
        except ValueError:
            self.hien_thi_loi("Dữ liệu phải là số!")

    # ========================================================
    # CLOSE EVENT
    # ========================================================
    def closeEvent(self, event):
        try:
            self.ngat_ket_noi_camera()
            self.ngat_ket_noi_camera_2()
            self.ngat_ket_noi_serial()
        except Exception:
            pass
        event.accept()


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================
if __name__ == "__main__":
    print("[INFO] OpenCV:", cv2.__version__)
    app = QtWidgets.QApplication(sys.argv)
    window = RobotControlApp()
    window.show()
    sys.exit(app.exec())
