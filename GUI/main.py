import sys
import os
import csv
import cv2
import torch
import numpy as np
from collections import Counter
from statistics import mode
import mysql.connector
import datetime
import time
import platform

from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QLineEdit, QMessageBox,
    QSizePolicy, QDialog, QDialogButtonBox, QHeaderView, QFileDialog, QComboBox,
    QSlider, QColorDialog, QFormLayout, QTextEdit, QGridLayout
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QCoreApplication, QSize
)
from PyQt5.QtGui import (
    QFont, QPixmap, QImage, QColor, QIcon, QTextCursor
)

import chatbotGPT
from chatbotUI import Chatbot

# -------------------------------------------------------------------------
                        # Global Config / Paths
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
bot_icon_path = os.path.join(BASE_DIR, "bot.png")

class Config:
    FONT_NAME = "Segoe UI"
    HEADER_FONT_SIZE = 24
    LABEL_FONT_SIZE = 12
    BUTTON_FONT_SIZE = 13

    HEADER_FONT = QFont(FONT_NAME, HEADER_FONT_SIZE, QFont.Bold)
    LABEL_FONT = QFont(FONT_NAME, LABEL_FONT_SIZE)
    BUTTON_FONT = QFont(FONT_NAME, BUTTON_FONT_SIZE, QFont.Bold)

    RESOLUTION = (1280, 720)

    DEFAULT_CLASS_COLORS = {
        "CDB": (255, 0, 0),    # Red
        "BMS": (255, 0, 255),  # Magenta
        "Maker": (0, 0, 139)   # Dark Blue
    }

    BUTTON_STYLES = {
         "left": """
            QPushButton {
                background-color: #D3D3D3;
                color: black;
                border: 1px solid #888;
                border-radius: 5px;
                padding: 8px 10px;
                min-width: 60px;
            }
            QPushButton:pressed {
                background-color: #B0B0B0;
            }
         """,
         "right": """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 1px solid #388E3C;
                border-radius: 5px;
                padding: 8px 10px;
                min-width: 60px;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
         """,
         "logout": """
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: 1px solid #C0392B;
                border-radius: 5px;
                padding: 8px 10px;
                min-width: 60px;
            }
            QPushButton:pressed {
                background-color: #C0392B;
            }
         """
    }

# -------------------------------------------------------------------------
                # Enumerate Cameras (Generic Approach)
# -------------------------------------------------------------------------
def enumerate_cameras(max_test=10):
    """
    Attempts to open cameras from index 0 to max_test-1.
    Returns a list of (camera_name, index) for each valid camera.
    Uses the same backend as later used in initialize_webcam.
    """
    camera_list = []
    for i in range(max_test):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            camera_name = f"Camera #{i}"
            camera_list.append((camera_name, i))
            cap.release()
    return camera_list

# -------------------------------------------------------------------------
                # Helper Classes
# -------------------------------------------------------------------------
class LoadingSplashScreen(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        w = int(screen_geometry.width() * 0.3)
        h = int(screen_geometry.height() * 0.3)
        self.setFixedSize(w, h)
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        self.logo_label = QLabel(self)
        pixmap = QPixmap(LOGO_PATH)
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        self.title_label = QLabel("Badmark Counting Software", self)
        self.title_label.setFont(Config.HEADER_FONT)
        self.title_label.setStyleSheet("color: black;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        self.loading_label = QLabel("Loading", self)
        self.loading_label.setFont(QFont(Config.FONT_NAME, 16))
        self.loading_label.setStyleSheet("color: black;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("User Login")
        self.resize(400, 250)
        self.setStyleSheet("background-color: #f0f0f0;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        prod_layout = QHBoxLayout()
        prod_label = QLabel("Product Type:")
        prod_label.setFont(Config.LABEL_FONT)
        self.product_input = QLineEdit()
        self.product_input.setPlaceholderText("Enter product type")
        prod_layout.addWidget(prod_label)
        prod_layout.addWidget(self.product_input)
        layout.addLayout(prod_layout)
        mark_layout = QHBoxLayout()
        mark_label = QLabel("Marklot:")
        mark_label.setFont(Config.LABEL_FONT)
        self.marklot_input = QLineEdit()
        self.marklot_input.setPlaceholderText("Enter marklot")
        mark_layout.addWidget(mark_label)
        mark_layout.addWidget(self.marklot_input)
        layout.addLayout(mark_layout)
        oper_layout = QHBoxLayout()
        oper_label = QLabel("Operator ID:")
        oper_label.setFont(Config.LABEL_FONT)
        self.operator_input = QLineEdit()
        self.operator_input.setPlaceholderText("Enter operator ID")
        oper_layout.addWidget(oper_label)
        oper_layout.addWidget(self.operator_input)
        layout.addLayout(oper_layout)
        self.product_input.returnPressed.connect(lambda: QTimer.singleShot(0, lambda: self.marklot_input.setFocus()))
        self.marklot_input.returnPressed.connect(lambda: QTimer.singleShot(0, lambda: self.operator_input.setFocus()))
        self.operator_input.returnPressed.connect(self.on_operator_return)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        common_style = """
            QPushButton {
                background-color: #757575;
                color: white;
                padding: 5px;
                border: 1px solid #5a5a5a;
                border-radius: 5px;
            }
        """
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        cancel_button = self.buttons.button(QDialogButtonBox.Cancel)
        ok_button.setStyleSheet(common_style)
        cancel_button.setStyleSheet(common_style)
        ok_button.setMinimumWidth(100)
        cancel_button.setMinimumWidth(100)
        ok_button.setAutoDefault(False)
        cancel_button.setAutoDefault(False)
        self.buttons.accepted.connect(self.manual_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.product_input.setFocus()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            if self.focusWidget() in (self.product_input, self.marklot_input):
                event.ignore()
                return
        super().keyPressEvent(event)

    def on_operator_return(self):
        if (self.product_input.text().strip() and 
            self.marklot_input.text().strip() and 
            self.operator_input.text().strip()):
            QTimer.singleShot(0, self.accept)

    def manual_accept(self):
        if (self.product_input.text().strip() and 
            self.marklot_input.text().strip() and 
            self.operator_input.text().strip()):
            self.accept()
        else:
            QMessageBox.warning(self, "Incomplete Data", "Please fill in all fields before proceeding.")

    def get_inputs(self):
        return (self.product_input.text().strip(),
                self.marklot_input.text().strip(),
                self.operator_input.text().strip())

class ModelLoader(QThread):
    model_loaded = pyqtSignal(object)
    def __init__(self, model_path, device):
        super().__init__()
        self.model_path = model_path
        self.device = device
    def run(self):
        model = YOLO(self.model_path).to(self.device)
        self.model_loaded.emit(model)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, conf_threshold=0.5, model_path="", camera_index=0,
                 csv_folder="", resolution=(1280, 720), class_colors=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(420, 320)
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.camera_index = camera_index
        self.csv_folder = csv_folder
        self.resolution = resolution
        self.class_colors = class_colors if class_colors else Config.DEFAULT_CLASS_COLORS.copy()
        self.camera_list = enumerate_cameras()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        conf_group = QGroupBox("Confidence")
        conf_group.setFont(Config.LABEL_FONT)
        conf_layout = QHBoxLayout(conf_group)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(conf_threshold * 100))
        self.conf_value_label = QLabel(f"{conf_threshold:.2f}")
        self.conf_value_label.setFont(Config.LABEL_FONT)
        self.conf_slider.valueChanged.connect(self.on_conf_slider)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        main_layout.addWidget(conf_group)
        color_group = QGroupBox("Class Colors")
        color_group.setFont(Config.LABEL_FONT)
        form_layout = QFormLayout(color_group)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(8)
        self.color_buttons = {}
        for cls_name in ["CDB", "BMS", "Maker"]:
            lbl = QLabel(cls_name)
            lbl.setFont(Config.LABEL_FONT)
            color_btn = QPushButton()
            color_btn.setFixedSize(60, 25)
            (r, g, b) = self.class_colors[cls_name]
            color_btn.setStyleSheet(f"background-color: rgb({r},{g},{b});")
            color_btn.clicked.connect(lambda _, c=cls_name: self.pick_color(c))
            form_layout.addRow(lbl, color_btn)
            self.color_buttons[cls_name] = color_btn
        main_layout.addWidget(color_group)
        param_group = QGroupBox("Other Settings")
        param_group.setFont(Config.LABEL_FONT)
        param_layout = QVBoxLayout(param_group)
        model_layout = QHBoxLayout()
        model_label = QLabel("YOLO Model:")
        model_label.setFont(Config.LABEL_FONT)
        self.model_line = QLineEdit(model_path)
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_line)
        model_layout.addWidget(model_browse_btn)
        cam_layout = QHBoxLayout()
        cam_label = QLabel("Camera:")
        cam_label.setFont(Config.LABEL_FONT)
        self.camera_combo = QComboBox()
        self.index_map = {}
        for (cam_name, idx) in self.camera_list:
            self.camera_combo.addItem(cam_name)
            self.index_map[cam_name] = idx
        chosen_name = None
        for (cam_name, idx) in self.camera_list:
            if idx == self.camera_index:
                chosen_name = cam_name
                break
        if chosen_name is None and len(self.camera_list) > 0:
            chosen_name = self.camera_list[0][0]
        if chosen_name:
            self.camera_combo.setCurrentText(chosen_name)
        cam_layout.addWidget(cam_label)
        cam_layout.addWidget(self.camera_combo)
        csv_layout = QHBoxLayout()
        csv_label = QLabel("CSV Folder:")
        csv_label.setFont(Config.LABEL_FONT)
        self.csv_line = QLineEdit(self.csv_folder)
        csv_browse_btn = QPushButton("Browse")
        csv_browse_btn.clicked.connect(self.browse_csv)
        csv_layout.addWidget(csv_label)
        csv_layout.addWidget(self.csv_line)
        csv_layout.addWidget(csv_browse_btn)
        res_layout = QHBoxLayout()
        res_label = QLabel("Resolution:")
        res_label.setFont(Config.LABEL_FONT)
        self.res_combo = QComboBox()
        self.res_combo.addItem("1280x720")
        self.res_combo.addItem("640x480")
        if self.resolution == (640, 480):
            self.res_combo.setCurrentText("640x480")
        else:
            self.res_combo.setCurrentText("1280x720")
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.res_combo)
        param_layout.addLayout(model_layout)
        param_layout.addLayout(cam_layout)
        param_layout.addLayout(csv_layout)
        param_layout.addLayout(res_layout)
        main_layout.addWidget(param_group)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept_settings)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

    def on_conf_slider(self, value):
        c = value / 100.0
        self.conf_value_label.setText(f"{c:.2f}")

    def pick_color(self, cls_name):
        current_rgb = self.class_colors[cls_name]
        dlg = QColorDialog(QColor(*current_rgb), self)
        if dlg.exec_() == QColorDialog.Accepted:
            chosen = dlg.selectedColor()
            r, g, b, _ = chosen.getRgb()
            self.class_colors[cls_name] = (r, g, b)
            btn = self.color_buttons[cls_name]
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "Model Files (*.pt *.pth)")
        if path:
            self.model_line.setText(path)

    def browse_csv(self):
        folder = QFileDialog.getExistingDirectory(self, "Select CSV Folder")
        if folder:
            self.csv_line.setText(folder)

    def accept_settings(self):
        try:
            self.conf_threshold = float(self.conf_value_label.text())
        except ValueError:
            self.conf_threshold = 0.5
        self.model_path = self.model_line.text().strip()
        chosen_cam_name = self.camera_combo.currentText()
        chosen_index = self.index_map.get(chosen_cam_name, 0)
        self.camera_index = chosen_index
        self.csv_folder = self.csv_line.text().strip()
        val = self.res_combo.currentText()
        if val == "640x480":
            self.resolution = (640, 480)
        else:
            self.resolution = (1280, 720)
        self.accept()

    def get_settings(self):
        return (self.conf_threshold, self.model_line.text().strip(),
                self.camera_index, self.csv_line.text().strip(),
                self.resolution, self.class_colors)

# -------------------------------------------------------------------------
                # Main GUI
# -------------------------------------------------------------------------
class ObjectCounterGUI(QMainWindow):
    def __init__(self, splash=None):
        super().__init__()
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.first_model_load = True
        self.splash = splash
        self.create_database_and_table()
        self.product_type = ""
        self.marklot = ""
        self.operator_id = ""
        try:
            self.db_connection = mysql.connector.connect(
                host="localhost", 
                port=3306, 
                user="", 
                password="", 
                database=""
            )
            print("[INFO] Database connection successful: sony")
        except mysql.connector.Error as err:
            print("[ERROR] Failed to connect to database:", err)
            self.db_connection = None
        self.setWindowTitle("Object Counting Software")
        screen_geo = QApplication.primaryScreen().availableGeometry()
        sw, sh = screen_geo.width(), screen_geo.height()
        w = int(sw * 0.8)
        h = int(sh * 0.8)
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.setGeometry(x, y, w, h)
        self.cap = None
        self.current_cam_index = 0
        self.model = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.sliding_window = None
        self.update_mode_count = 60
        self.last_class_modes = {0:0, 1:0, 2:0}
        self.running = False
        self.row_number = 1
        self.class_id_map = {0: "BMS", 1: "CDB", 2: "Maker"}
        self.resolution = Config.RESOLUTION
        self.last_frame = None
        self.confidence_threshold = 0.5
        self.desired_fps = 30
        self.model_path = MODEL_PATH
        self.csv_folder = os.path.join(os.path.expanduser("~"), "Desktop")
        self.class_colors = {
            "CDB": Config.DEFAULT_CLASS_COLORS["CDB"],
            "BMS": Config.DEFAULT_CLASS_COLORS["BMS"],
            "Maker": Config.DEFAULT_CLASS_COLORS["Maker"]
        }
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.fps = 0
        self.model_loader = ModelLoader(MODEL_PATH, self.device)
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.start()
        print("[INFO] Model loading started in background...")
        self.setup_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.setStyleSheet(f"""
            QGroupBox {{
                font: bold 14px '{Config.FONT_NAME}';
                color: #333333;
                border: 2px solid #607D8B;
                border-radius: 5px;
                margin-top: 20px;
                padding-top: 15px;
            }}
            QPushButton {{
                font: bold 13px '{Config.FONT_NAME}';
            }}
        """)
        print("[INFO] GUI Initialized.")
        self.initialize_device()
        self.initialize_webcam()

    def create_database_and_table(self):
        try:
            conn = mysql.connector.connect(host="localhost", port=3306, user="sony", password="sony")
            cursor = conn.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS sony")
            conn.commit()
            conn.database = "sony"
            create_table_query = """
                CREATE TABLE IF NOT EXISTS object_counts (
                    #####
                )
            """
            cursor.execute(create_table_query)
            conn.commit()
            cursor.close()
            conn.close()
            print("[INFO] Database and table checked/created successfully.")
        except mysql.connector.Error as err:
            print("[ERROR] Failed to create database/table:", err)

    def on_model_loaded(self, model):
        self.model = model
        print("[INFO] Model loaded and ready for detection.")
        if self.first_model_load:
            self.first_model_load = False
            if self.splash:
                self.splash.close()
            self.show_login_dialog()

    def initialize_device(self):
        if self.device == 'cuda:0':
            print("[INFO] CUDA available. Using GPU (cuda:0).")
        else:
            torch.set_num_threads(4)
            print(f"[INFO] CUDA not available. Using CPU with {torch.get_num_threads()} threads.")
        print(f"[INFO] Running on device: {self.device}")

    def initialize_webcam(self):
        # First, test if the desired camera is available
        if platform.system() == "Windows":
            test_cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_DSHOW)
        else:
            test_cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_V4L2)
        if not test_cap.isOpened():
            print(f"[ERROR] No webcam found on index {self.current_cam_index}.")
            raise RuntimeError("No webcam found. Please connect a webcam.")
        test_cap.release()
        # Now open with the chosen backend
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.current_cam_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        print(f"[INFO] Webcam initialized on index {self.current_cam_index} with resolution: {self.resolution}")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        header_layout = QHBoxLayout()
        tl_label = QLabel("TL:")
        tl_label.setFont(Config.LABEL_FONT)
        self.tl_field = QLineEdit("01")
        self.tl_field.setFont(Config.LABEL_FONT)
        self.tl_field.setStyleSheet("background-color: #F5F5F5;")
        self.tl_field.setMaxLength(2)
        self.tl_field.setFixedWidth(40)
        header_layout.addWidget(tl_label)
        header_layout.addWidget(self.tl_field)
        header_layout.addSpacing(20)
        prod_label = QLabel("Product Type:")
        prod_label.setFont(Config.LABEL_FONT)
        self.product_line = QLineEdit()
        self.product_line.setReadOnly(True)
        self.product_line.setStyleSheet("background-color: #F5F5F5;")
        self.product_line.setFont(Config.LABEL_FONT)
        marklot_label = QLabel("Marklot:")
        marklot_label.setFont(Config.LABEL_FONT)
        self.marklot_line = QLineEdit()
        self.marklot_line.setReadOnly(True)
        self.marklot_line.setStyleSheet("background-color: #F5F5F5;")
        self.marklot_line.setFont(Config.LABEL_FONT)
        oper_label = QLabel("Operator ID:")
        oper_label.setFont(Config.LABEL_FONT)
        self.operator_line = QLineEdit()
        self.operator_line.setReadOnly(True)
        self.operator_line.setStyleSheet("background-color: #F5F5F5;")
        self.operator_line.setFont(Config.LABEL_FONT)
        header_layout.addWidget(prod_label)
        header_layout.addWidget(self.product_line)
        header_layout.addSpacing(20)
        header_layout.addWidget(marklot_label)
        header_layout.addWidget(self.marklot_line)
        header_layout.addSpacing(20)
        header_layout.addWidget(oper_label)
        header_layout.addWidget(self.operator_line)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFont(Config.BUTTON_FONT)
        self.settings_btn.setStyleSheet(Config.BUTTON_STYLES["left"])
        self.settings_btn.clicked.connect(self.open_settings)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.settings_btn)
        main_layout.addLayout(header_layout)
        content_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        badmarks_group = QGroupBox("badmarks")
        badmarks_group.setFont(Config.LABEL_FONT)
        badmarks_layout = QHBoxLayout(badmarks_group)
        badmarks_layout.setSpacing(15)
        self.cdb_box = self.create_color_box("CDB", self.class_colors["CDB"])
        self.bms_box = self.create_color_box("BMS", self.class_colors["BMS"])
        self.maker_box = self.create_color_box("Maker", self.class_colors["Maker"])
        badmarks_layout.addLayout(self.cdb_box)
        badmarks_layout.addLayout(self.bms_box)
        badmarks_layout.addLayout(self.maker_box)
        left_panel.addWidget(badmarks_group)
        table_group = QGroupBox("Data Table")
        table_group.setFont(Config.LABEL_FONT)
        table_layout = QVBoxLayout(table_group)
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels(["No.", "CDB", "BMS", "Maker", "Total", "Action"])
        self.data_table.setRowCount(0)
        self.data_table.verticalHeader().setVisible(False)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.data_table)
        left_panel.addWidget(table_group)
        content_layout.addLayout(left_panel, 35)
        right_panel = QVBoxLayout()
        preview_group = QGroupBox("Object Preview")
        preview_group.setFont(Config.LABEL_FONT)
        preview_layout = QGridLayout(preview_group)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(5)
        self.preview_area = QLabel()
        self.preview_area.setAlignment(Qt.AlignCenter)
        self.preview_area.setStyleSheet("background-color: #000000;")
        preview_layout.addWidget(self.preview_area, 0, 0)
        preview_layout.setRowStretch(0, 1)
        preview_layout.setColumnStretch(0, 1)
        self.chatbot_btn = QPushButton()
        self.chatbot_btn.setFixedSize(40, 40)
        self.chatbot_btn.setIcon(QIcon(bot_icon_path))
        self.chatbot_btn.setIconSize(QSize(40, 40))
        self.chatbot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                border-radius: 20px;
            }
            QPushButton:pressed {
                background-color: #388E3C;
            }
        """)
        self.chatbot_btn.clicked.connect(self.open_chatbot_window)
        preview_layout.addWidget(self.chatbot_btn, 0, 0, Qt.AlignBottom | Qt.AlignRight)
        right_panel.addWidget(preview_group)
        content_layout.addLayout(right_panel, 65)
        main_layout.addLayout(content_layout)
        bottom_control = QHBoxLayout()
        bottom_control.setSpacing(15)
        left_controls = QHBoxLayout()
        left_controls.setSpacing(10)
        self.add_btn = QPushButton("Add")
        self.add_btn.setFont(Config.BUTTON_FONT)
        self.add_btn.setStyleSheet(Config.BUTTON_STYLES["left"])
        self.add_btn.clicked.connect(self.update_table)
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setFont(Config.BUTTON_FONT)
        self.upload_btn.setStyleSheet(Config.BUTTON_STYLES["left"])
        self.upload_btn.clicked.connect(self.upload_to_db)
        left_controls.addWidget(self.add_btn)
        left_controls.addWidget(self.upload_btn)
        bottom_control.addLayout(left_controls)
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(Config.LABEL_FONT)
        bottom_control.addWidget(self.fps_label)
        bottom_control.addStretch()
        self.toggle_btn = QPushButton("Start")
        self.toggle_btn.setFont(Config.BUTTON_FONT)
        self.toggle_btn.setStyleSheet(Config.BUTTON_STYLES["right"])
        self.toggle_btn.clicked.connect(self.toggle_detection)
        self.logout_btn = QPushButton("Logout")
        self.logout_btn.setFont(Config.BUTTON_FONT)
        self.logout_btn.setStyleSheet(Config.BUTTON_STYLES["logout"])
        self.logout_btn.clicked.connect(self.logout)
        right_controls = QHBoxLayout()
        right_controls.setSpacing(10)
        right_controls.addWidget(self.toggle_btn)
        right_controls.addSpacing(10)
        right_controls.addWidget(self.logout_btn)
        bottom_control.addLayout(right_controls)
        main_layout.addLayout(bottom_control)

    def create_color_box(self, label_text, rgb_tuple):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(label_text)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(Config.LABEL_FONT)
        layout.addWidget(title_label)
        r, g, b = rgb_tuple
        box = QLabel("0")
        box.setStyleSheet(f"background-color: rgba({r},{g},{b},128); border: 2px solid rgb({r},{g},{b});")
        box.setAlignment(Qt.AlignCenter)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        box.setFixedHeight(75)
        box.setFont(QFont(Config.FONT_NAME, 24, QFont.Bold))
        layout.addWidget(box)
        return layout

    def open_settings(self):
        # Pause detection if running
        if self.running:
            self.toggle_detection()

        # Store the current camera index and release the camera
        previous_cam_index = self.current_cam_index
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        # Open the settings dialog with the previous camera index
        dlg = SettingsDialog(
            parent=self,
            conf_threshold=self.confidence_threshold,
            model_path=self.model_path,
            camera_index=previous_cam_index,
            csv_folder=self.csv_folder,
            resolution=self.resolution,
            class_colors=self.class_colors
        )

        result = dlg.exec_()
        if result == QDialog.Accepted:
            # Update settings if accepted
            (new_conf, new_model, new_cam_idx, new_csv, new_res, new_colors) = dlg.get_settings()
            self.confidence_threshold = new_conf
            self.model_path = new_model
            self.csv_folder = new_csv
            self.class_colors = new_colors
            self.current_cam_index = new_cam_idx
            self.resolution = new_res
            self.initialize_webcam()
            
            # Reload model if path changed
            if new_model and new_model != self.model_path:
                print("[INFO] Reloading model from new path:", new_model)
                self.model_loader = ModelLoader(new_model, self.device)
                self.model_loader.model_loaded.connect(self.on_model_loaded)
                self.model_loader.start()
            
            self.update_color_boxes()
        else:
            # Revert to previous camera if cancelled
            self.current_cam_index = previous_cam_index
            self.initialize_webcam()

        # Remain paused after dialog closes
        self.running = False
        self.toggle_btn.setText("Start")
        self.toggle_btn.setStyleSheet(Config.BUTTON_STYLES["right"])

    def update_color_boxes(self):
        r, g, b = self.class_colors["CDB"]
        cdb_lbl = self.cdb_box.itemAt(1).widget()
        cdb_lbl.setStyleSheet(f"background-color: rgba({r},{g},{b},128); border: 2px solid rgb({r},{g},{b});")
        r, g, b = self.class_colors["BMS"]
        bms_lbl = self.bms_box.itemAt(1).widget()
        bms_lbl.setStyleSheet(f"background-color: rgba({r},{g},{b},128); border: 2px solid rgb({r},{g},{b});")
        r, g, b = self.class_colors["Maker"]
        maker_lbl = self.maker_box.itemAt(1).widget()
        maker_lbl.setStyleSheet(f"background-color: rgba({r},{g},{b},128); border: 2px solid rgb({r},{g},{b});")

    def open_chatbot_window(self):
        if self.running:
            print("[INFO] Pausing detection before opening chatbot.")
            self.toggle_detection()
        self.chatbot_window = Chatbot()
        self.chatbot_window.show()

    def toggle_detection(self):
        if not self.running:
            print("[INFO] Starting detection...")
            if self.model is None:
                QMessageBox.information(self, "Loading Model", "Please wait, the model is still loading...")
                return
            if self.cap is None:
                self.initialize_webcam()
            self.sliding_window = {0: [], 1: [], 2: []}
            interval_ms = int(1000 / self.desired_fps)
            self.timer.start(interval_ms)
            self.toggle_btn.setText("Stop")
            self.toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #B71C1C;
                    color: white;
                    border: 1px solid #880E4F;
                    border-radius: 5px;
                    padding: 8px 10px;
                    min-width: 60px;
                }
            """)
            self.running = True
            print("[INFO] Detection started.")
        else:
            print("[INFO] Pausing detection...")
            self.timer.stop()
            self.toggle_btn.setText("Start")
            self.toggle_btn.setStyleSheet(Config.BUTTON_STYLES["right"])
            self.running = False
            print("[INFO] Detection paused.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_C:
            self.switch_camera()
        else:
            super().keyPressEvent(event)

    def switch_camera(self):
        print("[INFO] Switching camera (next index)...")
        if self.running:
            self.toggle_detection()
        if self.cap:
            self.cap.release()
        new_idx = self.current_cam_index + 1
        if new_idx > 9:
            new_idx = 0
        self.current_cam_index = new_idx
        self.initialize_webcam()

    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            print("[WARNING] Camera not available.")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("[WARNING] Failed to read frame from camera.")
            return
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update = now
            self.fps_label.setText(f"FPS: {self.fps}")
        processed_frame, class_modes = self.process_frame(frame)
        self.last_class_modes = class_modes
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_image)
        scaled_pix = pix.scaled(self.preview_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_area.setPixmap(scaled_pix)
        self.last_frame = scaled_pix
        self.update_colored_boxes(class_modes)

    def process_frame(self, frame):
        results = self.model(frame, conf=self.confidence_threshold)
        detections = results[0].boxes.data.cpu().numpy()
        frame_counts = Counter()
        for det in detections:
            cls_id = int(det[5])
            frame_counts[cls_id] += 1
            x1, y1, x2, y2 = map(int, det[:4])
            if cls_id == 0:
                color = self.class_colors["BMS"]
            elif cls_id == 1:
                color = self.class_colors["CDB"]
            else:
                color = self.class_colors["Maker"]
            (r, g, b) = color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
        class_modes = {}
        for cid in [0, 1, 2]:
            if cid not in self.sliding_window:
                self.sliding_window[cid] = []
            self.sliding_window[cid].append(frame_counts[cid])
            if len(self.sliding_window[cid]) > self.update_mode_count:
                self.sliding_window[cid].pop(0)
            try:
                class_modes[cid] = mode(self.sliding_window[cid])
            except:
                class_modes[cid] = 0
        return frame, class_modes

    def update_colored_boxes(self, class_modes):
        cdb_count = class_modes.get(1, 0)
        bms_count = class_modes.get(0, 0)
        maker_count = class_modes.get(2, 0)
        self.cdb_box.itemAt(1).widget().setText(str(cdb_count))
        self.bms_box.itemAt(1).widget().setText(str(bms_count))
        self.maker_box.itemAt(1).widget().setText(str(maker_count))

    def update_table(self):
        n = self.data_table.rowCount()
        if n > 0:
            first_item = self.data_table.item(n - 1, 0)
            if first_item and first_item.text() == "Total":
                self.data_table.removeRow(n - 1)
        try:
            cdb_text = self.cdb_box.itemAt(1).widget().text()
            bms_text = self.bms_box.itemAt(1).widget().text()
            maker_text = self.maker_box.itemAt(1).widget().text()
            cdb_count = int(cdb_text) if cdb_text.isdigit() else 0
            bms_count = int(bms_text) if bms_text.isdigit() else 0
            maker_count = int(maker_text) if maker_text.isdigit() else 0
        except Exception as e:
            print("[ERROR] Reading counts from color boxes:", e)
            cdb_count, bms_count, maker_count = 0, 0, 0
        total = bms_count + cdb_count + maker_count
        new_row_no = self.data_table.rowCount() + 1
        row = self.data_table.rowCount()
        self.data_table.insertRow(row)
        item_no = QTableWidgetItem(str(new_row_no))
        item_no.setFlags(item_no.flags() & ~Qt.ItemIsEditable)
        self.data_table.setItem(row, 0, item_no)
        item_cdb = QTableWidgetItem(str(cdb_count))
        item_cdb.setFlags(item_cdb.flags() & ~Qt.ItemIsEditable)
        self.data_table.setItem(row, 1, item_cdb)
        item_bms = QTableWidgetItem(str(bms_count))
        item_bms.setFlags(item_bms.flags() & ~Qt.ItemIsEditable)
        self.data_table.setItem(row, 2, item_bms)
        item_maker = QTableWidgetItem(str(maker_count))
        item_maker.setFlags(item_maker.flags() & ~Qt.ItemIsEditable)
        self.data_table.setItem(row, 3, item_maker)
        item_total = QTableWidgetItem(str(total))
        item_total.setFlags(item_total.flags() & ~Qt.ItemIsEditable)
        self.data_table.setItem(row, 4, item_total)
        trash_icon_path = os.path.join(BASE_DIR, "trash.png")
        trash_icon = QIcon(trash_icon_path)
        del_btn = QPushButton()
        del_btn.setIcon(trash_icon)
        del_btn.setIconSize(QSize(16, 16))
        del_btn.setFixedSize(30, 30)
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff476c;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #fa5a7e;
            }
        """)
        del_btn.clicked.connect(self.delete_table_row)
        del_container = QWidget()
        del_layout = QHBoxLayout(del_container)
        del_layout.setContentsMargins(0, 0, 0, 0)
        del_layout.addWidget(del_btn)
        del_layout.setAlignment(Qt.AlignCenter)
        self.data_table.setCellWidget(row, 5, del_container)
        self.reassign_table_row_numbers()
        self.update_summary_row()
        self.highlight_duplicate_rows()

    def update_summary_row(self):
        n = self.data_table.rowCount()
        if n > 0:
            item = self.data_table.item(n - 1, 0)
            if item and item.text() == "Total":
                self.data_table.removeRow(n - 1)
        n = self.data_table.rowCount()
        sum_cdb = sum(int(self.data_table.item(i, 1).text()) for i in range(n) if self.data_table.item(i, 1))
        sum_bms = sum(int(self.data_table.item(i, 2).text()) for i in range(n) if self.data_table.item(i, 2))
        sum_maker = sum(int(self.data_table.item(i, 3).text()) for i in range(n) if self.data_table.item(i, 3))
        sum_total = sum(int(self.data_table.item(i, 4).text()) for i in range(n) if self.data_table.item(i, 4))
        summary_row = self.data_table.rowCount()
        self.data_table.insertRow(summary_row)
        bold_font = QFont(Config.FONT_NAME, 12, QFont.Bold)
        total_item = QTableWidgetItem("Total")
        total_item.setFlags(total_item.flags() & ~Qt.ItemIsEditable)
        total_item.setFont(bold_font)
        self.data_table.setItem(summary_row, 0, total_item)
        cdb_item = QTableWidgetItem(str(sum_cdb))
        cdb_item.setFlags(cdb_item.flags() & ~Qt.ItemIsEditable)
        cdb_item.setFont(bold_font)
        self.data_table.setItem(summary_row, 1, cdb_item)
        bms_item = QTableWidgetItem(str(sum_bms))
        bms_item.setFlags(bms_item.flags() & ~Qt.ItemIsEditable)
        bms_item.setFont(bold_font)
        self.data_table.setItem(summary_row, 2, bms_item)
        maker_item = QTableWidgetItem(str(sum_maker))
        maker_item.setFlags(maker_item.flags() & ~Qt.ItemIsEditable)
        maker_item.setFont(bold_font)
        self.data_table.setItem(summary_row, 3, maker_item)
        total_col_item = QTableWidgetItem(str(sum_total))
        total_col_item.setFlags(total_col_item.flags() & ~Qt.ItemIsEditable)
        total_col_item.setFont(bold_font)
        self.data_table.setItem(summary_row, 4, total_col_item)
        clear_btn = QPushButton("X")
        clear_btn.setFixedSize(30, 30)
        clear_btn.setStyleSheet("background-color: #e74c3c; color: white; border: none; border-radius: 15px;")
        clear_btn.clicked.connect(self.clear_table)
        clear_container = QWidget()
        clear_layout = QHBoxLayout(clear_container)
        clear_layout.setContentsMargins(0, 0, 0, 0)
        clear_layout.addWidget(clear_btn)
        clear_layout.setAlignment(Qt.AlignCenter)
        self.data_table.setCellWidget(summary_row, 5, clear_container)

    def clear_table(self):
        self.data_table.setRowCount(0)

    def highlight_duplicate_rows(self):
        n = self.data_table.rowCount()
        if n == 0:
            return
        if self.data_table.item(n - 1, 0) and self.data_table.item(n - 1, 0).text() == "Total":
            end = n - 1
        else:
            end = n
        for i in range(end):
            for j in range(self.data_table.columnCount()):
                item = self.data_table.item(i, j)
                if item:
                    item.setBackground(Qt.white)
        i = 0
        while i < end - 1:
            group = [i]
            cdb_val = self.data_table.item(i, 1).text() if self.data_table.item(i, 1) else ""
            bms_val = self.data_table.item(i, 2).text() if self.data_table.item(i, 2) else ""
            maker_val = self.data_table.item(i, 3).text() if self.data_table.item(i, 3) else ""
            j = i + 1
            while j < end:
                cdb2 = self.data_table.item(j, 1).text() if self.data_table.item(j, 1) else ""
                bms2 = self.data_table.item(j, 2).text() if self.data_table.item(j, 2) else ""
                maker2 = self.data_table.item(j, 3).text() if self.data_table.item(j, 3) else ""
                if cdb2 == cdb_val and bms2 == bms_val and maker2 == maker_val:
                    group.append(j)
                    j += 1
                else:
                    break
            if len(group) >= 2:
                for r in group:
                    for col in range(self.data_table.columnCount()):
                        item = self.data_table.item(r, col)
                        if item:
                            item.setBackground(QColor(255, 255, 153))
            i = j

    def reassign_table_row_numbers(self):
        n = self.data_table.rowCount()
        if n > 0 and self.data_table.item(n - 1, 0) and self.data_table.item(n - 1, 0).text() == "Total":
            end = n - 1
        else:
            end = n
        for i in range(end):
            item = QTableWidgetItem(str(i + 1))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.data_table.setItem(i, 0, item)

    def delete_table_row(self):
        button = self.sender()
        if button:
            for row in range(self.data_table.rowCount()):
                cell_widget = self.data_table.cellWidget(row, 5)
                if cell_widget and cell_widget.findChild(QPushButton) == button:
                    self.data_table.removeRow(row)
                    break
            self.reassign_table_row_numbers()
            self.update_summary_row()
            self.highlight_duplicate_rows()

    def upload_to_db(self):
        # Determine the number of data rows (excluding "Total" row if present)
        n = self.data_table.rowCount()
        if n > 0 and self.data_table.item(n - 1, 0) and self.data_table.item(n - 1, 0).text() == "Total":
            data_rows = n - 1
        else:
            data_rows = n

        # Check if table is empty
        if data_rows == 0:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Empty Table")
            msgBox.setText("The table is empty. There is no data to upload.")
            msgBox.setStandardButtons(QMessageBox.Ok)
            for button in msgBox.buttons():
                button.setStyleSheet("background-color: #D3D3D3; color: black;")
            msgBox.exec_()
            return

        # Get TL value and increment it
        try:
            current_val = int(self.tl_field.text())
        except ValueError:
            current_val = 1
        tl_val = current_val
        self.tl_field.setText(f"{current_val + 1:02d}")

        # Get metadata
        product_type = self.product_line.text()
        marklot = self.marklot_line.text()
        operator_val = self.operator_line.text()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialize CSV data with header
        csv_data = []
        csv_header = ["No.", "CDB", "BMS", "Maker", "Total", "Operator", "Timestamp"]
        csv_data.append(csv_header)

        # Collect data from table and prepare for database insertion
        data_to_insert = []
        for row in range(data_rows):
            no = int(self.data_table.item(row, 0).text())
            cdb_val = int(self.data_table.item(row, 1).text())
            bms_val = int(self.data_table.item(row, 2).text())
            maker_val = int(self.data_table.item(row, 3).text())
            total_val = cdb_val + bms_val + maker_val

            # Append to CSV data
            csv_data.append([no, cdb_val, bms_val, maker_val, total_val, operator_val, timestamp])
            # Collect data for database insertion
            data_to_insert.append((no, product_type, marklot, tl_val, cdb_val, bms_val, maker_val, total_val, operator_val))

        # Attempt database insertion if connection exists
        if self.db_connection:
            try:
                cursor = self.db_connection.cursor()
                insert_query = """
                    INSERT INTO object_counts (no, product_type, marklot, tl, cdb, bms, maker, total, operator)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, data_to_insert)
                self.db_connection.commit()
                cursor.close()
                print(f"[INFO] Uploaded {data_rows} rows to the database.")
            except Exception as e:
                print("[ERROR] Failed to upload data to database:", e)
                QMessageBox.critical(self, "Database Upload Failed", f"Data upload to database failed: {e}\nData will still be saved to CSV.")
        else:
            print("[WARNING] No database connection available. Proceeding with CSV only.")

        # Save the CSV file with the collected data
        self.save_csv_to_desktop(product_type, marklot, tl_val, csv_data)
        QMessageBox.information(self, "Data Saved", f"{data_rows} rows processed. Data saved as CSV.")
        self.data_table.setRowCount(0)

    def save_csv_to_desktop(self, product_type, marklot, tl, csv_data):
        folder = self.csv_folder if self.csv_folder else os.path.join(os.path.expanduser("~"), "Desktop")
        now = datetime.datetime.now()
        year_folder = os.path.join(folder, now.strftime("%Y"))
        month_folder = os.path.join(year_folder, now.strftime("%m"))
        os.makedirs(month_folder, exist_ok=True)
        safe_product = product_type.replace(" ", "_")
        safe_marklot = marklot.replace(" ", "_")
        file_name = f"{safe_product}_{safe_marklot}_{tl:02d}.csv"
        file_path = os.path.join(month_folder, file_name)
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in csv_data:
                    writer.writerow(row)
            print(f"[INFO] CSV file saved: {file_path}")
        except Exception as e:
            print("[ERROR] Failed to save CSV file:", e)

    def show_login_dialog(self):
        login_dialog = LoginDialog()
        login_dialog.setWindowModality(Qt.ApplicationModal)
        if login_dialog.exec_() == QDialog.Accepted:
            self.product_type, self.marklot, self.operator_id = login_dialog.get_inputs()
            print(f"[INFO] Login details: Product Type: {self.product_type}, Marklot: {self.marklot}, Operator ID: {self.operator_id}")
            self.product_line.setText(self.product_type)
            self.marklot_line.setText(self.marklot)
            self.operator_line.setText(self.operator_id)
            self.tl_field.setText("01")
            self.show()
        else:
            print("[INFO] Login cancelled. Exiting application.")
            QCoreApplication.quit()

    def logout(self):
        if self.running:
            self.toggle_detection()
        login_dialog = LoginDialog()
        login_dialog.setWindowModality(Qt.ApplicationModal)
        if login_dialog.exec_() == QDialog.Accepted:
            self.product_type, self.marklot, self.operator_id = login_dialog.get_inputs()
            print(f"[INFO] New login details: {self.product_type}, {self.marklot}, {self.operator_id}")
            self.product_line.setText(self.product_type)
            self.marklot_line.setText(self.marklot)
            self.operator_line.setText(self.operator_id)
            self.tl_field.setText("01")
        else:
            print("[INFO] Logout cancelled; retaining current login.")

# -------------------------------------------------------------------------
                # Main Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(LOGO_PATH))
    splash = LoadingSplashScreen()
    splash.show()
    app.processEvents()
    window = ObjectCounterGUI(splash)
    sys.exit(app.exec_())