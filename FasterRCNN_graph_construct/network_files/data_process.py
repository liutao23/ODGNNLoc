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
BASE_PATH = r"E:\TTMNdata\University\test"
DRONE_PATH = os.path.join(BASE_PATH, "query_drone")
SATELLITE_PATH = os.path.join(BASE_PATH, "query_satellite")
# DRONE_PATH = os.path.join(BASE_PATH, "gallery_drone")
# SATELLITE_PATH = os.path.join(BASE_PATH, "gallery_satellite")
# NUM=14  #University

# NUM = 12 #300
# NUM = 11 #250
# NUM = 10 # 200
NUM = 9#150

# NUM = 12  #IR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 目标检测模型
def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model

# 加载模型
satellite_model = create_model(num_classes=1 + 1)  # 假设背景类已经被考虑在内
satellite_model.load_state_dict(torch.load(r'C:\Users\liutao\Desktop\TTMN_CGMN\pre_weights\new_fastercnn_building\SUES\resNetFpn-model-59(1).pth')["model"])
satellite_model = satellite_model.to(device)
satellite_model.eval()

# 加载模型
drone_model = create_model(num_classes=1 + 1)  # 假设背景类已经被考虑在内
drone_model.load_state_dict(torch.load(r'C:\Users\liutao\Desktop\TTMN_CGMN\pre_weights\new_fastercnn_building\SUES\resNetFpn-model-59(1).pth')["model"])
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

def clean_drone_folder(folder_path, model):
    """
    清理文件夹，保留检测到目标的图片
    :param folder_path: 文件夹路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 检测阈值
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if drone_detect_objects(file_path, model) < NUM:
                print(f'remove{file_path}')
                os.remove(file_path)


    # 删除空文件夹
    for root, dirs, _ in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                print(f'remove{dir_path}')
                os.rmdir(dir_path)

def clean_satellite_folder(folder_path, model):
    """
    清理文件夹，保留检测到目标的图片
    :param folder_path: 文件夹路径
    :param model: 检测模型
    :param device: 设备
    :param threshold: 检测阈值
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if drone_satellite_objects(file_path, model) < NUM:
                print(f'remove{file_path}')
                os.remove(file_path)

    # 删除空文件夹
    for root, dirs, _ in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                print(f'remove{dir_path}')
                os.rmdir(dir_path)
# 清理无人机和卫星图片
clean_drone_folder(DRONE_PATH, drone_model)
clean_satellite_folder(SATELLITE_PATH, satellite_model)

# 确保无人机和卫星子文件夹一致
drone_subdirs = set(os.listdir(DRONE_PATH))
satellite_subdirs = set(os.listdir(SATELLITE_PATH))

# 删除多余的卫星文件夹
extra_satellite_subdirs = satellite_subdirs - drone_subdirs
for subdir in extra_satellite_subdirs:
    print(f'remove{os.path.join(SATELLITE_PATH, subdir)}')
    shutil.rmtree(os.path.join(SATELLITE_PATH, subdir))

# 删除多余的无人机文件夹
extra_drone_subdirs = drone_subdirs - satellite_subdirs
for subdir in extra_drone_subdirs:
    print(f'remove{os.path.join(DRONE_PATH, subdir)}')
    shutil.rmtree(os.path.join(DRONE_PATH, subdir))
