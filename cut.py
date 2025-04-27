import cv2
import os

def extract_frames(video_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率：{fps} FPS")
    print(f"总帧数：{total_frames}")

    # 当前处理到的帧数
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 构建输出文件名，从00000.jpg开始
        frame_filename = os.path.join(output_folder, f"{frame_count:05d}.jpg")

        # 保存当前帧
        cv2.imwrite(frame_filename, frame)
        print(f"已保存：{frame_filename}")

        frame_count += 1

    # 释放资源
    cap.release()
    print(f"共提取了 {frame_count} 帧")

if __name__ == "__main__":
    # 输入视频文件路径
    video_path = "D:\HDU\STORE\sam2/notebooks/videos\demo2.mp4"  # 替换为你的视频文件路径

    # 输出帧存放的文件夹
    output_folder = "D:\HDU\STORE\sam2/notebooks/videos\demo2"

    extract_frames(video_path, output_folder)