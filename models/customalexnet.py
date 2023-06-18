import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import alexnet,resnet50
from transformers import BertModel
import torchvision.transforms as transforms
from transformers import BertTokenizer

# 定义图像模型
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        resnet = resnet50(pretrained=True)
        # 去掉最后的全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # 冻结卷积层的权重
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 新建一个全连接层
        self.fc = nn.Linear(resnet.fc.in_features, 512)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# 定义文本模型
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1386, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, text):
        features = self.fc(text)
        return features

# 定义图文互检模型
class CrossModalModel(nn.Module):
    def __init__(self, num_classes):
        super(CrossModalModel, self).__init__()
        self.image_model = ImageModel()
        self.text_model = TextModel()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images, text):
        image_features = self.image_model(images)
        text_features = self.text_model(text)
        #点乘，来衡量相关度
        features = image_features * text_features
        fuse_feature = features.view(features.size(0), -1)
        # 将图像特征向量和文本特征向量进行组合，得到一个融合后的特征向量。
        # 这个特征向量可以被送入后续的分类器或回归器中，
        # 以完成具体的任务，比如图像与文本的配对、图像分类、文本分类等。
        out = self.fc(features)
        return out,fuse_feature,image_features,text_features


class CustomAlexNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomAlexNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)
        self.num_classes = num_classes
        # 文本特征提取模块
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # 图文匹配模块
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_appended_layer(self):
        appended_layer = nn.Linear(
            self.model._modules['classifier'][4].out_features,
            self.num_classes
        )
        appended_layer.weight.data.normal_(0, 0.001)

        return appended_layer

    def forward(self, images, captions):
        # 图像特征提取
        image_features = self.resnet(images)
        # 文本特征提取
        input_ids = captions['input_ids']
        attention_mask = captions['attention_mask']
        text_features = self.bert(input_ids, attention_mask=attention_mask)[1]
        # 图文匹配
        features = torch.cat((image_features, text_features), dim=1)
        x = self.fc1(features)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = CustomAlexNet(24)
    img = Image.open('../data/mirflickr25k/im1.jpg')
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = image_transform(img)
    text = 'the cat is on the desk.'
    img = img.unsqueeze(0)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    res = model(img,tokens)
    # for key, value in model.named_modules():
    #     print(key, value)
