import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2  # 导入OpenCV库用于视频处理

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs
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

# 视频目录设置
video_dir = "D:\HDU\STORE\sam2/notebooks/videos\demo"

# 扫描所有JPEG帧文件名
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# 初始化推理状态
inference_state = predictor.init_state(video_path=video_dir)

# 用于存储点击点的字典
prompts = {}

# 处理第一个对象
ann_frame_idx = 0
ann_obj_id_yaocong = 2
def pro_1_with_point():
    points = np.array([[770, 370], [700, 390],[730,340],[820,360]], dtype=np.float32)  # 添加负点击点
    labels = np.array([1, 0, 0, 0], dtype=np.int32)  # 1 表示正点击，0 表示负点击
    prompts[ann_obj_id_yaocong] = points, labels

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_yaocong,
        points=points,
        labels=labels,
    )
def pro_1_with_box():

    # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
    box = np.array([740, 350, 800, 400], dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_yaocong,
        box=box,
    )




def pro_2_with_point():
    
    ann_frame_idx = 0
    ann_obj_id_hengtu = 3

    points = np.array([[535,450],[575, 400],[575,330],[575,400],[575, 375],[625, 395],[525, 375]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([0,0,0,0,1,0,0], np.int32)
    prompts[ann_obj_id_hengtu] = points, labels

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_hengtu,
        points=points,
        labels=labels,
    )


ann_obj_id_hengtu = 3 
def pro_2_with_box():
    ann_frame_idx = 0  # the frame index we interact with
      # give a unique id to each object we interact with (it can be any integers)

    # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
    box = np.array([540, 350, 600, 400], dtype=np.float32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_hengtu,
        box=box,
    )
# pro_1_with_box()
# pro_2_with_box()
pro_1_with_point()
pro_2_with_point()
pro_2_with_point()

# 在整个视频中传播分割结果
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


# 获取视频帧的尺寸信息
sample_frame = Image.open(os.path.join(video_dir, frame_names[0]))
frame_height, frame_width = sample_frame.height, sample_frame.width

# 打印帧的尺寸信息以进行调试
print(f"Frame dimensions: width = {frame_width}, height = {frame_height}")

# 设置视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output-pp.mp4', fourcc, 25.0, (frame_width, frame_height))

# 遍历每一帧并生成带有分割结果的视频
for out_frame_idx in range(len(frame_names)):
    frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
    frame = np.array(Image.open(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间以适应OpenCV

    if out_frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            # 确保掩码的形状与帧的形状匹配
            out_mask = out_mask.astype(np.bool_)
            out_mask_expanded = np.expand_dims(out_mask, axis=-1)  # 扩展掩码维度以匹配帧的形状

            # # 将掩码调整到与帧相同的大小
            # print(f"Mask shape: {out_mask_expanded.shape}, Frame shape: {(frame_height, frame_width)}")
            if out_mask_expanded.shape[1:3] != (frame_height, frame_width):
                out_mask_resized = cv2.resize(
                    out_mask.astype(np.uint8),
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST
                )
                out_mask_resized = np.expand_dims(out_mask_resized, axis=-1)
            else:
                out_mask_resized = out_mask_expanded

            # 将掩码应用到帧上
            mask_color = np.array([0, 255, 0])  # 绿色掩码
            if out_obj_id == ann_obj_id_yaocong:
                mask_color = np.array([0, 0, 255])  # 红色掩码用于第一个对象
            elif out_obj_id == ann_obj_id_hengtu:
                mask_color = np.array([255, 0, 0])  # 蓝色掩码用于第二个对象

            # print(f"Mask shape after resizing: {out_mask_resized.shape}")
            out_mask_resized = out_mask_resized.squeeze(0)
            out_mask_resized = out_mask_resized.squeeze(-1)
            
            frame[out_mask_resized] = frame[out_mask_resized] * 0.7 + mask_color * 0.3  # 半透明显示

    out_video.write(frame)

# 释放视频写入器
out_video.release()