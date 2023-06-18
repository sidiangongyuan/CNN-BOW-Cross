import time
from datasets.Mirflickr25kDataset import Mirflickr25kDataset
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mysql.connector
import torchvision.models as models
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request
import os
import re

from preprocess_mirflickr25k import get_mirflickr_tag_map
from models.customalexnet import CrossModalModel

app = Flask(
    __name__,
    static_folder='.',  # 表示为上级目录 (myproject/) 开通虚拟资源入口
    static_url_path='',  # 这是路径前缀, 个人认为非常蛋疼的设计之一, 建议传空字符串, 可以避免很多麻烦
)
mirflickr25k_data_dir = './data'
image_path_root = './data/mirflickr25k/'
# -------------------数据库连接操作
cnx = mysql.connector.connect(user='root', password='nb000000', database='myfirst', host='localhost')
cursor = cnx.cursor()
# ------------------- 读取模型操作
model = torch.load('save.pt')
# #加载resnet模型
# model = models.resnet50(pretrained=True)
# #nn.Identity()是PyTorch中的一个函数，可以创建一个无操作(即恒等操作)的模块。它不会改变输入，只是简单地将其输出。
# # 在神经网络的某些结构中，可以使用它来跳过一些层或在不改变特征大小的情况下传递输入。
# model.fc = nn.Identity()
# model.eval()
#图片的变化
transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#tag2id就是一个词典，决定了你输入文本的特征向量。id2tag存的是所有的annotation，也就是text
tag2id, id2tag = get_mirflickr_tag_map()
text_dim = len(id2tag)


#-----------------------------基本页面
@app.route('/', methods=['GET', 'POST'])
def index():
    #如果有东西传回来，那么就执行相应代码
    if request.method == 'POST':
        result_count = int(request.form.get('result_count'))
        if 'text' in request.form and request.form['text']!='':  # 假设前端提交的文本数据名称为'text'
            k = 0
            img_list = []
            tags_list = []
            # 进行文本数据的操作
            text_data = request.form['text']
            word_list = text_data.split()
            #vector是文本特征向量,
            vector = np.zeros((1386,1))
            # 遍历单词列表，将向量对应位置的值加1
            for word in word_list:
                # 查找单词在词袋库中的位置
                position = tag2id.get(word,-1)
                # 将向量对应位置的值加1
                vector[position] += 1
            res = search_similar_texts(vector,result_count)
            for x in res:
                # 0是路径，1是相似度，2是标签
                s = x[0]
                s = '.' + s
                # new_path = s.replace("../data/mirflickr25k/", "../data/image/")
                img_list.append(s)
                tags_list.append(x[2])
                #需要把zip这个函数传入给前端，才能识别zip这个函数
            return render_template('results1.html',image_urls=img_list, tags_list=tags_list, zip=zip)
        else:  # 假设前端提交的图像数据名称为'image'
            file = request.files['image']
            # 进行图像数据的操作
            # TODO:这个必须要放在这，不能放在全局。要不然是累加的图片，不会自己清空。
            img_list = []
            tags_list = []
            img = Image.open(file.stream)
            res = search_similar_images(img,result_count)
            for x in res:
                # 0是路径，1是相似度，2是标签
                s = x[0]
                s = '.' + s
                # new_path = s.replace("../data/mirflickr25k/", "../data/image/")
                img_list.append(s)
                tags_list.append(x[2])
                #TODO: zip  要传标签
            return render_template('results2.html', image_urls=img_list, tags_list=tags_list, zip=zip)  # 返回到指定网页， 使用render_template 那么默认就去templates 文件夹找

    return render_template('search.html')


#-----------------------------将文本、图片和融合的的所有特征存入Mysql ，能不能同时将对应的文本的内容也存入数据库呢？
def feature_Mysql():
    #出来有3个，，image label 和 text
    datasets = Mirflickr25kDataset(transform=transform)
    #这个for循环只会执行一遍
    for root, _, filenames in os.walk(mirflickr25k_data_dir + '/mirflickr25k/meta/tags'):
        filenames.sort(key=lambda x: int(x[4:-4]))
        for i,((image, label, text), filename) in enumerate(zip(datasets,filenames)):
            file = os.path.join(root, filename)
            with open(file, encoding='UTF-8') as f:
                # 把一张图片的tag全部读出，存入Lines，它是一个list
                lines = f.readlines()
                tags = list(map(lambda l: l.split()[0], lines))
                my_list_tag = ','.join(tags)
            #1 × 3 × 256 × 256
            image = image.unsqueeze(0)
            with torch.no_grad():
                #图片提取特征
                text = torch.tensor(text)
                text = text.unsqueeze(0)
                #传入的维度是 1× 图片维度   文本是 1×1386
                _,fuse_features,image_feature,text_feature = model(image,text)
                #出来的维度都是 1 × 512 
                fuse_features = fuse_features.numpy()
                image_feature = image_feature.numpy()
                text_feature = text_feature.numpy()
                #图片路径获取
                image_path = image_path_root + 'im' + str(i+1) + '.jpg'
                query = "INSERT INTO resnet50 (path,fuse_features,tag,image_feature,text_feature) VALUES (%s,%s,%s,%s,%s)"
                values = (image_path,fuse_features.tobytes(),my_list_tag,image_feature.tobytes(),text_feature.tobytes())
                cursor.execute(query, values)
    cnx.commit()
    cursor.close()
    cnx.close()
    print('success')


#-----------------------------文本的tag存入Mysql
def tag_mysql():
    #这个for循环只会执行一遍
    for root, _, filenames in os.walk(mirflickr25k_data_dir + '/mirflickr25k/meta/tags'):
        # x[4:-4]表示的是 数字，tag 的数字
        filenames.sort(key=lambda x: int(x[4:-4]))
        for i, filename in enumerate(filenames):
            file = os.path.join(root, filename)
            with open(file, encoding='UTF-8') as f:
                # 把一张图片的tag全部读出，存入Lines，它是一个list
                lines = f.readlines()
                tags = list(map(lambda l: l.split()[0], lines))
                my_list_tag = ','.join(tags)
                # # 将文本转换为词袋向量
                # vectorizer = CountVectorizer()
                # #传入文本list
                # text_matrix = vectorizer.fit_transform(tags)
                # # 获取文本向量列表
                # text_vectors = text_matrix.toarray()
                query = "INSERT INTO images (tag) VALUES (%s)"
                values = ([my_list_tag],)
                cursor.execute(query, values)

    cnx.commit()
    cursor.close()
    cnx.close()
    print('success')


#-----------------------------以文搜图
def search_similar_texts(text,k):
    #生成一个 1:3:256:256的Tensor，全1，来代表image
    image_tensor = torch.ones((1, 3, 256, 256))
    text = torch.tensor(text)
    text = text.T
    text = text.to(torch.float32)
    with torch.no_grad():
        _,_,_,text_features = model(image_tensor,text)
    # text_features = text_features.detach()
    # text_features = text_features.numpy()
    similarities = []
    query = ("SELECT path,fuse_features,tag,image_feature,text_feature FROM resnet50")
    cursor.execute(query)
    results = cursor.fetchall()
    for result in results:
        #文件名
        filename = result[0]
        #标签名
        tag = result[2]
        #用逗号分割开
        words = tag.split(',')
        text_feature_mysql = np.frombuffer(result[4], dtype=np.float32)
        similarity = np.dot(text_features, text_feature_mysql) / (np.linalg.norm(text_features) * np.linalg.norm(text_feature_mysql))
        similarities.append((filename, similarity,words))
    similarities.sort(key=lambda x: x[1], reverse=True)
        # for word in words:
        #     #TODO:子串，全部串，部分串，看看能不能识别,这样如何区别模糊的和精确的呢？可以将标签到时候也一起显示
        #     result = re.search(text,word)
        #     if result:
        #         temp_list.append(word)
        #     else:
        #         continue
        # if len(temp_list) != 0:
        #     tags_list.append(temp_list)
        #     img_list.append(filename)
    return similarities[:k]


#-----------------------------输入查询图像并检索相似的图像
def search_similar_images(img, k=10):
    # 提取查询图像的特征
    # query_image = Image.open(img)
    vector = np.ones((1386, 1))
    text = torch.tensor(vector)
    text = text.T
    text = text.to(torch.float32)
    query_feature_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out,fuse_feature,image_feature,text_feature = model(query_feature_tensor,text)
    # 构建查询语句
    query = ("SELECT path,fuse_features,tag,image_feature,text_feature FROM resnet50")
    cursor.execute(query)
    results = cursor.fetchall()
    # 计算相似度并返回前k个最相似的图像
    similarities = []
    for result in results:
        #文件名
        filename = result[0]
        #标签
        tags = result[2]
        words = tags.split(',')
        # np.frombuffer 是用来将一个二进制数据缓冲区转化成一个 NumPy 数组
        feature = np.frombuffer(result[3], dtype=np.float32)
        similarity = np.dot(image_feature, feature) / (np.linalg.norm(image_feature) * np.linalg.norm(feature))
        similarities.append((filename, similarity,words))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[1:k+1]



if __name__ == '__main__':
    # feature_Mysql()
    # path_list = []
    app.run(debug=True)
    #PIL格式的img
    # query_image = Image.open('./data/mirflickr25k/im1.jpg')
    # r = search_similar_images(query_image)
    # for x in r:
    #     path_list.append(x[0])
    # print(path_list)


