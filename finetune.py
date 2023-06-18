from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.Mirflickr25kDataset import Mirflickr25kDataset, split_dataset
from models.customalexnet import CustomAlexNet,CrossModalModel



def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, label, tags) in enumerate(train_loader):
        output = model(data,tags)
        # output = torch.softmax(output,dim=1)
        loss = criterion(output, label)
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader), loss.item()),
                  end=' === ')
            print('with learning rate:', optimizer.param_groups[0]['lr'])


@torch.no_grad()
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for data, label, text in test_loader:
        output = model(data,text)
        # output = torch.softmax(output,dim=1)
        test_loss += criterion(output, label).item()
        pred = output.argmax(1)

        un_one_hot_label = label.argmax(1)
        correct += pred.eq(un_one_hot_label.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


def finetune():
    batch_size = 32
    epochs = 25
    #用cpu
    device = torch.device('cuda')

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Mirflickr25kDataset(transform=transform)
    # 分割训练和测试
    train_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = CrossModalModel(24)

    # optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0005, lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    point = 0
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch=epoch)
        acc = test(model, device, test_loader, criterion)
        if acc > point:
            point = acc
            torch.save(model,'save.pt')
            print("save success!")
if __name__ == '__main__':
    finetune()
