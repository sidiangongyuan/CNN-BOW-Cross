from PIL import Image
import os


def crop_images(source_folder, target_folder, size=(256, 256)):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取源文件夹下所有图片文件的路径
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(source_folder, image_file)
        image = Image.open(image_path)

        # 裁剪图片
        image.thumbnail(size)

        image.save(os.path.join(target_folder, image_file))

    print("缩放完成！")


# 调用示例
source_folder = "./data/mirflickr25k"  # 源文件夹路径
target_folder = "./data/image"  # 目标文件夹路径
crop_images(source_folder, target_folder)
