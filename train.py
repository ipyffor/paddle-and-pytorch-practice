import torch
from torchvision import transforms
from loader import get_loader
from trainer import Trainer

EPOCHS = 200
BATCH_SIZE = 128
LR = 0.01
model_path = './results/model'
sets_path = './data_sets'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def data_loader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = get_loader(BATCH_SIZE, shuffle=True, num_workers=2, transform = transform_train, train=True)
    test_loader = get_loader(BATCH_SIZE, shuffle=False, num_workers=2, transform=transform_test, train=False)
    return train_loader, test_loader

train_loader, test_loader = data_loader()
trainer = Trainer(LR, device, model_path)

def train(epoch):
    loss_sum = 0.
    correct = 0.
    total = 0.
    for id, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)
        loss,lohits = trainer.forward(images, labels)
        loss_sum += loss.item()

        ## 计算准确率
        _, predict = torch.max(lohits, 1)
        correct += predict.eq(labels).detach().cpu().sum().item()
        total += labels.size(0)

        ## 打印
        avg_loss = loss_sum/id
        acc = correct/total
        # print(correct, total)
        print('Train ==> Epoch: {}, loss: {}, acc: {}'.format(epoch, avg_loss, acc))
        ## 更新
        trainer.train_update()
    trainer.save_model(fname='resnet_epoch{}.pth'.format(epoch))
    pass

def evaluation():
    correct = 0
    total = 0
    for id, (images, labels) in enumerate(test_loader, 1):
        images, labels = images.to(device), labels.to(device)
        loss, logits = trainer.forward(images, labels, is_eval=True)
        _, predict = torch.max(logits, 1)
        correct += predict.eq(labels).sum().item()
        total += labels.size(0)
    acc = correct/total
    print('Evalute ===> acc: {}'.format(acc))
    pass

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train(epoch)
        evaluation()