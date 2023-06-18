from PIL import Image
import torch
from torchvision.models import resnet50
from transformers import BertModel
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import BertTokenizer
import torch.nn as nn
from torchvision import models

def test():
    # 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 待处理的文本
    text = "This is a girl who is smoking."
    # 对文本进行分词和编码
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])



    image_model = resnet50(pretrained=True)
    # 加载预训练BERT模型
    bert = BertModel.from_pretrained('bert-base-uncased')

    # 设置模型为评估模式
    image_model.eval()
    bert.eval()

    # 获取文本对应的词嵌入向量
    with torch.no_grad():
        outputs = bert(input_ids)
        #outputs[0]为每个token的隐藏状态，
        # 维度为(batch_size, sequence_length, hidden_size)，
        # outputs[1]为CLS token的隐藏状态，维度为(batch_size, hidden_size)。
        #由于我们只需要提取语义向量，所以只需要[0]
        text_emb = outputs[0][:, 0, :]

    # 加载图像和文本数据
    image = Image.open('./data/mirflickr25k/im1.jpg')
    # 图像预处理
    image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)


    class ImageEncoder(nn.Module):
        def __init__(self, img_size=224, embed_size=512):
            super(ImageEncoder, self).__init__()
            self.resnet = models.resnet50(pretrained=True)
            self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.resnet.fc = nn.Linear(2048, embed_size)

        def forward(self, x):
            x = x.unsqueeze(0)
            x = self.resnet(x)
            return x


    class TextEncoder(nn.Module):
        def __init__(self, embed_size=512):
            super(TextEncoder, self).__init__()
            self.fc = nn.Linear(768, embed_size)

        def forward(self, x):
            x = self.fc(x)
            return x


    class CrossModalEmbedding(nn.Module):
        def __init__(self, img_size=224, embed_size=512):
            super(CrossModalEmbedding, self).__init__()
            self.img_encoder = ImageEncoder(img_size, embed_size)
            self.text_encoder = TextEncoder(embed_size)

        def forward(self, img, text):
            img_emb = self.img_encoder(img)
            text_emb = self.text_encoder(text)
            return img_emb, text_emb

    cross = CrossModalEmbedding()
    img_emb, text_emb = cross(image,text_emb)
    #计算相似度
    similarity = F.cosine_similarity(img_emb, text_emb)
    print(similarity)
