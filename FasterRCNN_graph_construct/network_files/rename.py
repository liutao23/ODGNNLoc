#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2025/1/16 3:52
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author : Tao Liu
# Unit：Nanjing University of Science and Technology
# @Time : 2025/1/5 19:21

path = 'E:/TTMNdata/University/train'
import os

# 指定起始数字，使用整数来避免前导零问题
start_number = 0

for file in os.listdir(path):

    # 指定要重命名的父文件夹路径
    parent_folder_path = path + '/' + file

    # 获取父文件夹中所有子文件夹的名称列表
    subfolders = sorted(os.listdir(parent_folder_path), key=lambda x: int(x) if x.isdigit() else float('inf'))

    # 重命名子文件夹
    for index, subfolder in enumerate(subfolders, start=start_number):
        # 构建当前子文件夹的完整路径
        current_path = os.path.join(parent_folder_path, subfolder)

        # 检查当前路径是否是一个文件夹（排除文件）
        if os.path.isdir(current_path):
            # 构建新的文件夹名称，使用四位数的格式
            new_name = f'{index:04d}'  # 使用四位数的格式

            # 构建新的文件夹完整路径
            new_path = os.path.join(parent_folder_path, new_name)

            # 重命名文件夹
            os.rename(current_path, new_path)
            print(f'Renamed: {current_path} -> {new_path}')

    print('Renaming completed.')