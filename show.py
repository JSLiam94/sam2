import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QFileDialog, QSlider, QLineEdit, QComboBox, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QTimer
import torch
from PIL import Image
import matplotlib.pyplot as plt
import time

from PIL import Image, ImageDraw, ImageFont

import datetime
import csv  # 导入 csv 模块

# SAM2 导入和初始化代码保持不变
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

log_file = 'log.txt'

with open(log_file, 'a', encoding='UTF-8') as f:
    # 日期时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"{current_time} - SAM2 导入和初始化代码保持不变\n")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "D:\HDU\STORE\sam2\sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# 用于显示掩码、点和框的辅助函数
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


class VideoAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频标注与 SAM2 分割工具")
        self.setGeometry(100, 100, 600, 400)

        self.video_dir = ""  
        self.frame_names = []  
        self.frames = []
        self.current_frame_idx = 0

        self.positive_points = []  
        self.negative_points = []  
        self.bounding_boxes = []   

        self.prompts = {}  
        self.inference_state = None  

        self.current_mode = None  # 当前模式
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

        self.current_ann_obj_id = 1  # 当前选择的 ann_obj_id
        self.ann_obj_ids = [1]  # 可供选择的 ann_obj_id 列表

        self.sam_flag = False

        # 输出视频相关
        self.out_video = None
        self.output_video_path = "output.mp4"
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_height = 0
        self.frame_width = 0

        self.init_ui()
        self.setup_events()

    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # 按钮布局（横向）
        self.controls_layout = QHBoxLayout()

        self.load_button = QPushButton("加载图片文件夹", self)
        self.controls_layout.addWidget(self.load_button)

        self.play_button = QPushButton("播放/暂停", self)
        self.controls_layout.addWidget(self.play_button)

        self.prev_frame_button = QPushButton("上一帧", self)
        self.controls_layout.addWidget(self.prev_frame_button)

        self.next_frame_button = QPushButton("下一帧", self)
        self.controls_layout.addWidget(self.next_frame_button)

        self.add_positive_button = QPushButton("添加正点击点", self)
        self.controls_layout.addWidget(self.add_positive_button)

        self.add_negative_button = QPushButton("添加负点击点", self)
        self.controls_layout.addWidget(self.add_negative_button)

        self.add_box_button = QPushButton("添加框", self)
        self.controls_layout.addWidget(self.add_box_button)

        self.run_sam_button = QPushButton("运行 SAM2 分割", self)
        self.controls_layout.addWidget(self.run_sam_button)

        # 添加删除当前 ann_obj_id 的按钮
        self.delete_ann_obj_id_button = QPushButton("删除当前 ann_obj_id", self)
        self.controls_layout.addWidget(self.delete_ann_obj_id_button)

        # 添加保存为 CSV 的按钮
        self.save_csv_button = QPushButton("保存为 CSV", self)
        self.controls_layout.addWidget(self.save_csv_button)

        # 添加从 CSV 加载的按钮
        self.load_csv_button = QPushButton("从 CSV 加载", self)
        self.controls_layout.addWidget(self.load_csv_button)

        # 添加 ann_obj_id 的下拉菜单
        self.ann_obj_id_combo = QComboBox(self)
        self.ann_obj_id_combo.addItems([str(obj_id) for obj_id in self.ann_obj_ids])
        self.controls_layout.addWidget(self.ann_obj_id_combo)

        # 添加按钮以添加新的 ann_obj_id
        self.add_ann_obj_id_button = QPushButton("添加新的 ann_obj_id", self)
        self.controls_layout.addWidget(self.add_ann_obj_id_button)

        # 添加输出视频按钮
        self.output_video_button = QPushButton("输出视频", self)
        self.controls_layout.addWidget(self.output_video_button)

        self.layout.addLayout(self.controls_layout)

        # 滑块单独放在一行
        self.slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider_layout.addWidget(self.slider)
        self.layout.addLayout(self.slider_layout)

        # 消息框单独放在一行
        self.log_layout = QVBoxLayout()
        self.log_text = QLineEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setText("状态：未加载图片文件夹")
        self.log_layout.addWidget(self.log_text)
        self.layout.addLayout(self.log_layout)

        self.main_widget.setLayout(self.layout)

    def setup_events(self):
        self.load_button.clicked.connect(self.load_image_folder)
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.add_positive_button.clicked.connect(self.set_positive_point_mode)
        self.add_negative_button.clicked.connect(self.set_negative_point_mode)
        self.add_box_button.clicked.connect(self.set_box_mode)
        self.run_sam_button.clicked.connect(self.run_sam2)
        self.output_video_button.clicked.connect(self.output_video)
        self.slider.valueChanged.connect(self.slider_changed)
        self.add_ann_obj_id_button.clicked.connect(self.add_new_ann_obj_id)
        self.delete_ann_obj_id_button.clicked.connect(self.delete_current_ann_obj_id)
        self.save_csv_button.clicked.connect(self.save_to_csv)
        self.load_csv_button.clicked.connect(self.load_from_csv)
        self.ann_obj_id_combo.currentIndexChanged.connect(self.on_ann_obj_id_changed)

        self.video_label.mousePressEvent = self.mouse_press_event
        self.video_label.mouseReleaseEvent = self.mouse_release_event
        self.video_label.mouseMoveEvent = self.mouse_move_event

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

    def on_ann_obj_id_changed(self):
        # 更新当前的 ann_obj_id
        print('now',self.ann_obj_id_combo.currentText())
        self.current_ann_obj_id = int(self.ann_obj_id_combo.currentText())
        self.log_text.setText(f"已选择 ann_obj_id: {self.current_ann_obj_id}")
        self.draw_on_frame()  # 刷新显示

    def load_image_folder(self):
        self.video_dir = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
        if self.video_dir:
            self.frame_names = [
                p for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

            self.frames = []
            for frame_name in self.frame_names:
                frame_path = os.path.join(self.video_dir, frame_name)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)

            if self.frames:
                self.current_frame_idx = 0
                self.slider.setMaximum(len(self.frames) - 1)
                self.slider.setValue(0)
                self.show_frame(0)
                self.log_text.setText(f"状态：已加载图片文件夹，共 {len(self.frames)} 帧")

                # 初始化 SAM2 推理状态
                self.inference_state = predictor.init_state(video_path=self.video_dir)

                # 获取视频帧的尺寸信息
                sample_frame = Image.open(os.path.join(self.video_dir, self.frame_names[0]))
                self.frame_height, self.frame_width = sample_frame.height, sample_frame.width

    def show_frame(self, frame_idx):
        if 0 <= frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(1000)  

    def play_video(self):
        if self.frames:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
            self.slider.setValue(self.current_frame_idx)
            self.show_frame(self.current_frame_idx)

    def prev_frame(self):
        if self.frames:
            self.current_frame_idx = (self.current_frame_idx - 1) % len(self.frames)
            self.slider.setValue(self.current_frame_idx)
            self.show_frame(self.current_frame_idx)

    def next_frame(self):
        if self.frames:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
            self.slider.setValue(self.current_frame_idx)
            self.show_frame(self.current_frame_idx)

    def slider_changed(self, value):
        if self.frames:
            self.current_frame_idx = value
            self.show_frame(self.current_frame_idx)

    def mouse_press_event(self, event):
        if self.frames:
            x = event.pos().x()
            y = event.pos().y()
            width = self.video_label.width()
            height = self.video_label.height()
            frame_width = self.frames[0].shape[1]
            frame_height = self.frames[0].shape[0]
            x_ratio = x / width
            y_ratio = y / height
            x_pixel = int(x_ratio * frame_width)
            y_pixel = int(y_ratio * frame_height)

            if self.current_mode == "添加正点击点":
                self.positive_points.append((x_pixel, y_pixel, self.current_ann_obj_id))
                self.log_text.setText(f"添加正点击点到 ann_obj_id {self.current_ann_obj_id}: ({x_pixel}, {y_pixel})")
                with open(log_file, "a", encoding='UTF-8') as f:
                    f.write(f"添加正点击点到 ann_obj_id {self.current_ann_obj_id}: ({x_pixel}, {y_pixel})\n")

            elif self.current_mode == "添加负点击点":
                self.negative_points.append((x_pixel, y_pixel, self.current_ann_obj_id))
                self.log_text.setText(f"添加负点击点到 ann_obj_id {self.current_ann_obj_id}: ({x_pixel}, {y_pixel})")
                with open(log_file, "a", encoding='UTF-8') as f:
                    f.write(f"添加负点击点到 ann_obj_id {self.current_ann_obj_id}: ({x_pixel}, {y_pixel})\n")

            elif self.current_mode == "添加框":
                self.start_x, self.start_y = x_pixel, y_pixel
                self.end_x, self.end_y = x_pixel, y_pixel

            self.draw_on_frame()

    def mouse_move_event(self, event):
        if self.current_mode == "添加框" and self.frames:
            x = event.pos().x()
            y = event.pos().y()
            width = self.video_label.width()
            height = self.video_label.height()
            frame_width = self.frames[0].shape[1]
            frame_height = self.frames[0].shape[0]
            x_ratio = x / width
            y_ratio = y / height
            self.end_x = int(x_ratio * frame_width)
            self.end_y = int(y_ratio * frame_height)
            self.draw_on_frame()

    def mouse_release_event(self, event):
        if self.current_mode == "添加框" and self.frames:
            self.end_x = max(self.start_x, self.end_x)
            self.end_y = max(self.start_y, self.end_y)
            self.bounding_boxes.append((self.start_x, self.start_y, self.end_x, self.end_y, self.current_ann_obj_id))
            self.log_text.setText(f"添加框到 ann_obj_id {self.current_ann_obj_id}: ({self.start_x}, {self.start_y}, {self.end_x}, {self.end_y})")
            with open(log_file, "a", encoding='UTF-8') as f:
                f.write(f"添加框到 ann_obj_id {self.current_ann_obj_id}: ({self.start_x}, {self.start_y}, {self.end_x}, {self.end_y})\n")
            self.draw_on_frame()

    def set_positive_point_mode(self):
        self.current_mode = "添加正点击点"
        self.log_text.setText("模式：添加正点击点")

    def set_negative_point_mode(self):
        self.current_mode = "添加负点击点"
        self.log_text.setText("模式：添加负点击点")

    def set_box_mode(self):
        self.current_mode = "添加框"
        self.log_text.setText("模式：添加框")

    def add_new_ann_obj_id(self):
        new_id = max(self.ann_obj_ids) + 1
        self.ann_obj_ids.append(new_id)
        self.ann_obj_id_combo.addItem(str(new_id))  # 添加新项到下拉菜单
        self.ann_obj_id_combo.setCurrentIndex(len(self.ann_obj_ids)-1)  # 设置新项为当前选中项
        self.current_ann_obj_id = new_id  # 同步更新当前 ann_obj_id
        self.log_text.setText(f"已添加新的 ann_obj_id: {new_id}")

    def delete_current_ann_obj_id(self):
        # 删除当前 current_ann_obj_id 的所有点和框
        self.positive_points = [point for point in self.positive_points if point[2] != self.current_ann_obj_id]
        self.negative_points = [point for point in self.negative_points if point[2] != self.current_ann_obj_id]
        self.bounding_boxes = [box for box in self.bounding_boxes if box[4] != self.current_ann_obj_id]
        self.log_text.setText(f"已删除 ann_obj_id {self.current_ann_obj_id} 的所有点和框")
        predictor.reset_state(self.inference_state)
        self.draw_on_frame()  # 刷新显示

    def save_to_csv(self):
        # 保存正负点击点和框的信息到 CSV 文件
        csv_file = QFileDialog.getSaveFileName(self, "保存为 CSV 文件", "", "CSV 文件 (*.csv)")[0]
        if csv_file:
            with open(csv_file, 'w', newline='', encoding='UTF-8') as f:
                writer = csv.writer(f)
                # 写入标题行
                writer.writerow(["Type", "X", "Y",'b3','b4', "Ann_Obj_ID"])
                # 写入正点击点
                for x, y, obj_id in self.positive_points:
                    writer.writerow(["Positive", x, y,0,0,obj_id])
                # 写入负点击点
                for x, y, obj_id in self.negative_points:
                    writer.writerow(["Negative", x, y,0,0,obj_id])
                # 写入框
                for box in self.bounding_boxes:
                    x1, y1, x2, y2, obj_id = box
                    writer.writerow(["Box", x1, y1, x2, y2, obj_id])
            self.log_text.setText(f"数据已保存到 {csv_file}")

    def load_from_csv(self):
        # 从 CSV 文件加载正负点击点和框的信息
        csv_file = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV 文件 (*.csv)")[0]
        if csv_file:
            self.positive_points = []
            self.negative_points = []
            self.bounding_boxes = []
            loaded_obj_ids = set()  # 用于收集加载的 obj_id

            with open(csv_file, 'r', encoding='UTF-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    if row[0] == "Positive":
                        x = int(row[1])
                        y = int(row[2])
                        obj_id = int(row[5])
                        self.positive_points.append((x, y, obj_id))
                        loaded_obj_ids.add(obj_id)
                    elif row[0] == "Negative":
                        x = int(row[1])
                        y = int(row[2])
                        obj_id = int(row[5])
                        self.negative_points.append((x, y, obj_id))
                        loaded_obj_ids.add(obj_id)
                    elif row[0] == "Box":
                        x1 = int(row[1])
                        y1 = int(row[2])
                        x2 = int(row[3])
                        y2 = int(row[4])
                        obj_id = int(row[5])
                        self.bounding_boxes.append((x1, y1, x2, y2, obj_id))
                        loaded_obj_ids.add(obj_id)

            # # 更新 ann_obj_ids 列表
            # self.ann_obj_ids = sorted(list(loaded_obj_ids)) if loaded_obj_ids else [1]
            # self.ann_obj_id_combo.clear()
            # for i in range(len(self.ann_obj_ids)):
            #     #是数字才添加
            #     if isinstance(self.ann_obj_ids[i], int):
            #         self.ann_obj_id_combo.addItem(str(self.ann_obj_ids[i]))
            # # self.current_ann_obj_id = self.ann_obj_ids[0] if self.ann_obj_ids else 1

            self.log_text.setText(f"数据已从 {csv_file} 加载")
            self.draw_on_frame()  # 刷新显示
    def draw_on_frame(self):
        if self.frames:
            frame = self.frames[self.current_frame_idx].copy()
            # 绘制正点击点
            current_positive_points = [(x, y) for x, y, obj_id in self.positive_points if obj_id == self.current_ann_obj_id]
            for x, y in current_positive_points:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 绘制负点击点
            current_negative_points = [(x, y) for x, y, obj_id in self.negative_points if obj_id == self.current_ann_obj_id]
            for x, y in current_negative_points:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # 绘制框
            current_boxes = [(box[0], box[1], box[2], box[3]) for box in self.bounding_boxes if box[4] == self.current_ann_obj_id]
            for box in current_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

            # 如果在添加框模式下，绘制当前框
            if self.current_mode == "添加框":
                cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 0, 0), 2)

            self.show_frame_with_annotation(frame)

    def show_frame_with_annotation(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def run_sam2(self):
        self.sam_flag = True
        if not self.frames or not self.inference_state:
            self.log_text.setText("错误：未加载图片文件夹或 SAM2 初始化失败")
            return

        ann_frame_idx = self.current_frame_idx

        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[ann_frame_idx])))

        # 分别处理每个 ann_obj_id
        for obj_id in self.ann_obj_ids:
            current_positive_points = [(x, y) for x, y, oid in self.positive_points if oid == obj_id]
            current_negative_points = [(x, y) for x, y, oid in self.negative_points if oid == obj_id]
            current_boxes = [(box[0], box[1], box[2], box[3]) for box in self.bounding_boxes if box[4] == obj_id]

            if current_positive_points or current_negative_points:
                points = np.array(current_positive_points + current_negative_points, dtype=np.float32)
                labels = np.array([1]*len(current_positive_points) + [0]*len(current_negative_points), dtype=np.int32)

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

                show_points(points, labels, plt.gca())

            elif current_boxes:
                box = np.array(current_boxes[-1], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box,
                )

                show_box(box, plt.gca())

            else:
                continue

            for i, out_oid in enumerate(out_obj_ids):
                if out_oid == obj_id:
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    show_mask(mask, plt.gca(), obj_id=obj_id)

        plt.show()
        self.log_text.setText("状态：SAM2 分割完成")
    def output_video(self):

        self.font_path = 'simsun.ttc'
        self.font_size = 20
        if os.path.exists(self.font_path):
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        else:
            print(f"警告：未找到字体文件 {self.font_path}，可能无法正确显示中文文字。")
        # 在类的初始化方法中定义一个颜色映射
        self.mask_colors = {
            1: np.array([0, 255, 0]),    # 绿色掩码用于 ann_obj_id 1
            2: np.array([0, 0, 255]),    # 红色掩码用于 ann_obj_id 2
            3: np.array([255, 0, 0]),    # 蓝色掩码用于 ann_obj_id 3
            4: np.array([255, 255, 0]),  # 黄色掩码用于 ann_obj_id 4
            5: np.array([255, 0, 255]),  # 紫色掩码用于 ann_obj_id 5
            6: np.array([0, 255, 255]),
            7: np.array([0, 100, 100]),
            8: np.array([128, 255, 0]),
            9: np.array([128, 0, 255]),
            10: np.array([128, 128, 255]),
            11: np.array([128, 128, 0]),
            12: np.array([0, 128, 255])
        }
        # 创建一个字典，用于存储每个对象对应的标签
        # self.area_dict = {
        #     1: '腰丛', 
        #     2: '横突', 
        #     3: '腰大肌', 
        #     4: '椎体', 
        #     5: '腹主动静脉', 
        #     6: '竖脊肌', 
        #     7: '腰方肌', 
        #     8: '腹外斜肌', 
        #     9: '腹内斜肌',
        #     10: '腹横肌',
        #     11: 'yang',
        #     12: 'yang'
        # }
        self.area_dict = {
            1: '', 
            2: '', 
            3: '', 
            4: '', 
            5: '', 
            6: '竖脊肌', 
            7: '腰方肌', 
            8: '腹外斜肌', 
            9: '腹内斜肌',
            10: '腹横肌',
            11: 'yang',
            12: 'yang'
        }

        if self.sam_flag == False:
            self.log_text.setText("先运行SAM2")
            return

        if not self.frames or not self.inference_state:
            self.log_text.setText("错误：未加载图片文件夹或 SAM2 初始化失败")
            return

        # 初始化视频写入器
        self.out_video = cv2.VideoWriter(self.output_video_path, self.fourcc, 25.0, (self.frame_width, self.frame_height))

        # 在整个视频中传播分割结果
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # 遍历每一帧并生成带有分割结果的视频
        for out_frame_idx in range(len(self.frame_names)):
            frame_path = os.path.join(self.video_dir, self.frame_names[out_frame_idx])
            frame = np.array(Image.open(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间以适应 OpenCV

            # 如果有分割结果，应用掩码到帧上
            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    # 确保掩码的形状与帧的形状匹配
                    out_mask = out_mask.astype(np.bool_)
                    out_mask_expanded = np.expand_dims(out_mask, axis=-1)  # 扩展掩码维度以匹配帧的形状

                    # 将掩码调整到与帧相同的大小
                    if out_mask_expanded.shape[1:3] != (self.frame_height, self.frame_width):
                        out_mask_resized = cv2.resize(
                            out_mask.astype(np.uint8),
                            (self.frame_width, self.frame_height),
                            interpolation=cv2.INTER_NEAREST
                        )
                        out_mask_resized = np.expand_dims(out_mask_resized, axis=-1)
                    else:
                        out_mask_resized = out_mask_expanded

                    # 根据 ann_obj_id 选择掩码颜色
                    mask_color = self.mask_colors.get(out_obj_id, np.array([0, 255, 0]))

                    out_mask_resized = out_mask_resized.squeeze(0)
                    out_mask_resized = out_mask_resized.squeeze(-1)
                    
                    frame[out_mask_resized] = frame[out_mask_resized] * 0.7 + mask_color * 0.3  # 半透明显示

                    # 计算掩码的中心位置
                    coords = np.argwhere(out_mask_resized)
                    if len(coords) > 0:
                        center = np.mean(coords, axis=0).astype(int)
                        # 获取标签
                        label = self.area_dict.get(out_obj_id, f"Object {out_obj_id}")
                        
                        # 使用PIL绘制中文文字
                        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        if self.font:
                            draw.text((center[1], center[0]), label, fill=(255, 255, 255), font=self.font)
                        else:
                            draw.text((center[1], center[0]), label, fill=(255, 255, 255))
                        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 写入帧到视频文件
            if self.out_video:
                self.out_video.write(frame)

        # 释放视频写入器
        if self.out_video:
            self.out_video.release()
            self.log_text.setText(f"状态：视频已输出到 {self.output_video_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnnotator()
    window.show()
    sys.exit(app.exec_())