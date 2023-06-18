from datasets.Mirflickr25kDataset import Mirflickr25kDataset
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mysql.connector
import torchvision.models as models

def main():
    image_path_root = './data/mirflickr25k/'
    # -------------------数据库连接操作
    cnx = mysql.connector.connect(user='root', password='nb000000',database='myfirst',host='localhost')
    cursor = cnx.cursor()
    #执行语句
    # mycursor.execute("SELECT * FROM images")
    # myresult = mycursor.fetchall()
    # 输出查询结果
    # for x in myresult:
    #     print(x)
    # ------------------- 读取模型操作
    #加载resnet模型
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    #图片的变化
    transform = transforms.Compose([
        #把数据转换为tensfroms格式
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    #出来有3个，，image label 和 text
    datasets = Mirflickr25kDataset(transform = transform)
    for i, (image, image_path, caption) in enumerate(datasets):
        image = image.unsqueeze(0)
        with torch.no_grad():
            #提取特征
            features = model(image).squeeze().numpy()
            image_path = image_path_root + 'im' + str(i+1) + '.jpg'
            query = "INSERT INTO images (path, features) VALUES (%s, %s)"
            values = (image_path, features.tobytes())
            cursor.execute(query, values)

    cnx.commit()
    cursor.close()
    cnx.close()
    print('success')

if __name__ == '__main__':
    main()
