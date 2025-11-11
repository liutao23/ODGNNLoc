import os
import shutil
from PIL import Image
import torch
from torchvision import transforms
from backbone import resnet50_fpn_backbone
from network_files import FasterRCNN
import warnings

warnings.filterwarnings("ignore", message=".*torch.meshgrid:.*")

# 配置路径
BASE_PATH = r"D:\BaiduNetdiskDownload\SUES\Testing\300"
# DRONE_PATH = os.path.join(BASE_PATH, "drone")
# SATELLITE_PATH = os.path.join(BASE_PATH, "satellite")
# DRONE_PATH = os.path.join(BASE_PATH, "query_drone")
# SATELLITE_PATH = os.path.join(BASE_PATH, "query_satellite")
DRONE_PATH = os.path.join(BASE_PATH, "gallery_drone")
SATELLITE_PATH = os.path.join(BASE_PATH, "gallery_satellite")

# 设置目标文件夹
DRONE_DEST_PATH = r"C:\Users\liutao\Desktop\SUES\300\train_all\train_all\drone"
SATELLITE_DEST_PATH = r"C:\Users\liutao\Desktop\SUES\300\train_all\train_all\satellite"

# a = DRONE_DEST_PATH[:-5]
# b = SATELLITE_DEST_PATH[:-9]
# 创建存放删除日志的文件夹路径
drone_log_path = DRONE_DEST_PATH[:-5] + "drone_remove_log.txt"
satellite_log_path = SATELLITE_DEST_PATH[:-9]  + "satellite_remove_log.txt"

NUM = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 目标检测模型
def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model

# 加载模型
satellite_model = create_model(num_classes=17 + 1)  # 假设背景类已经被考虑在内
satellite_model.load_state_dict(torch.load(r'C:\Users\liutao\Desktop\ODGC\pre_weights\FasterRCNN17\share_detect.pth')["model"])
satellite_model = satellite_model.to(device)
satellite_model.eval()

# 加载模型
drone_model = create_model(num_classes=17 + 1)  # 假设背景类已经被考虑在内
drone_model.load_state_dict(torch.load(r'C:\Users\liutao\Desktop\ODGC\pre_weights\FasterRCNN17\share_detect.pth')["model"])
drone_model = drone_model.to(device)
drone_model.eval()

def load_image(path):
    data_transform = transforms.Compose([transforms.ToTensor()])
    # 读取无人机图像
    img = Image.open(path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0).to(device)
    # 检查是否为全零张量
    if torch.all(img == 0):
        print(f"无效图像{path}")
        pass
    return img

def drone_detect_objects(image_path, model):
    """
    检测图片中的目标数量
    :param image_path: 图片路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 置信度阈值
    :return: 检测到的目标数量
    """
    image = load_image(image_path)
    with torch.no_grad():
        drone_result = drone_model(image)
    # 根据置信度筛选目标
    satellite_box_num = drone_result[0]['boxes'].size(0)
    num_objects = satellite_box_num
    return num_objects

def drone_satellite_objects(image_path, model):
    """
    检测图片中的目标数量
    :param image_path: 图片路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 置信度阈值
    :return: 检测到的目标数量
    """
    image = load_image(image_path)
    with torch.no_grad():
        satellite_result = satellite_model(image)
    # 根据置信度筛选目标
    satellite_box_num = satellite_result[0]['boxes'].size(0)
    num_objects = satellite_box_num
    return num_objects

def copy_drone_folder(folder_path, dest_folder, model, remove_log):
    """
    复制检测到目标的图片到新的文件夹，并记录删除的文件夹路径
    :param folder_path: 文件夹路径
    :param dest_folder: 目标文件夹路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 检测阈值
    :param remove_log: 删除的文件夹路径日志
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if drone_detect_objects(file_path, model) >= NUM:
                dest_path = os.path.join(dest_folder, os.path.relpath(file_path, folder_path))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)
                print(f"复制 {file_path} 到 {dest_path}")

    # 删除空文件夹并记录日志
    for root, dirs, _ in os.walk(dest_folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                print(f"删除空文件夹 {dir_path}")
                remove_log.append(dir_path)
                os.rmdir(dir_path)

def copy_satellite_folder(folder_path, dest_folder, model, remove_log):
    """
    复制检测到目标的图片到新的文件夹，并记录删除的文件夹路径
    :param folder_path: 文件夹路径
    :param dest_folder: 目标文件夹路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 检测阈值
    :param remove_log: 删除的文件夹路径日志
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if drone_satellite_objects(file_path, model) >= NUM:
                dest_path = os.path.join(dest_folder, os.path.relpath(file_path, folder_path))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)
                print(f"复制 {file_path} 到 {dest_path}")

    # 删除空文件夹并记录日志
    for root, dirs, _ in os.walk(dest_folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                print(f"删除空文件夹 {dir_path}")
                remove_log.append(dir_path)
                os.rmdir(dir_path)

# 创建删除日志
drone_remove_log = []
satellite_remove_log = []

# 复制无人机和卫星图片
copy_drone_folder(DRONE_PATH, DRONE_DEST_PATH, drone_model, drone_remove_log)
copy_satellite_folder(SATELLITE_PATH, SATELLITE_DEST_PATH, satellite_model, satellite_remove_log)

# 保存删除的文件夹路径到txt文件
def save_removed_paths(log_path, remove_log):
    """
    保存删除的路径到日志文件
    :param log_path: 日志文件路径
    :param remove_log: 删除的路径列表
    """
    # 如果日志文件已经存在，读取它的内容并附加
    with open(log_path, "a") as f:
        for path in remove_log:
            f.write(path + "\n")

# 确保无人机和卫星子文件夹一致
drone_subdirs = set(os.listdir(DRONE_DEST_PATH))
satellite_subdirs = set(os.listdir(SATELLITE_DEST_PATH))



# 初始化删除记录列表
drone_remove_log = []
satellite_remove_log = []

# 删除多余的卫星文件夹并记录日志
extra_satellite_subdirs = satellite_subdirs - drone_subdirs
for subdir in extra_satellite_subdirs:
    extra_path = os.path.join(SATELLITE_DEST_PATH, subdir)
    print(f"删除多余的卫星文件夹 {extra_path}")
    shutil.rmtree(extra_path)
    satellite_remove_log.append(extra_path)  # 记录删除的路径

# 删除多余的无人机文件夹并记录日志
extra_drone_subdirs = drone_subdirs - satellite_subdirs
for subdir in extra_drone_subdirs:
    extra_path = os.path.join(DRONE_DEST_PATH, subdir)
    print(f"删除多余的无人机文件夹 {extra_path}")
    shutil.rmtree(extra_path)
    drone_remove_log.append(extra_path)  # 记录删除的路径

# 保存删除的路径到日志文件
save_removed_paths(drone_log_path, drone_remove_log)
save_removed_paths(satellite_log_path, satellite_remove_log)
