import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot
# matplotlib.use('Agg')
#如果网络能在GPU中训练，就使用GPU；否则使用CPU进行训练
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 定义训练集和测试集的不同预处理
train_transform = torchvision.transforms.Compose([
    # 训练集专用增强
    torchvision.transforms.RandomRotation(10),      # 随机旋转 ±10 度
    torchvision.transforms.RandomAffine(           # 随机平移 + 其他增强
        degrees=0,                  # 不额外旋转（已单独处理旋转）
        translate=(0.1, 0.1),       # 水平和垂直方向随机平移 10%
        scale=(0.9, 1.1),            # 随机缩放 90%~110%
        shear=5                      # 随机剪切 ±5 度
    ),
    # 基础预处理
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

# 测试集不需要数据增强
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

path = './mnist_dataset/'

# 应用不同的预处理
trainData = torchvision.datasets.MNIST(
    path,
    train=True,
    transform=train_transform,  # 训练集用增强版
    download=True
)
testData = torchvision.datasets.MNIST(
    path,
    train=False,
    transform=test_transform    # 测试集用基础版
)

#设定每一个Batch的大小
BATCH_SIZE = 256

#构建数据集和测试集的DataLoader
trainDataLoader = torch.utils.data.DataLoader(dataset = trainData,batch_size = BATCH_SIZE,shuffle = True)
testDataLoader = torch.utils.data.DataLoader(dataset = testData,batch_size = BATCH_SIZE)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            # The size of the picture is 28x28
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            #torch.nn.Softmax(dim=1)
        )


    def forward(self, input):
        output = self.model(input)
        return output
net = Net()
#将模型转换到device中，并将其结构显示出来
print(net.to(device))
lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 50
# 存储训练过程
history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    net.train(True)
    for step, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        labels = labels.to(device)

        net.zero_grad()
        outputs = net(trainImgs)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        #optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            net.train(False)
            for testImgs, labels in testDataLoader:
                testImgs = testImgs.to(device)
                labels = labels.to(device)
                outputs = net(testImgs)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss / len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()
# 对测试Loss进行可视化
matplotlib.pyplot.figure(figsize=(10, 5))  # 明确指定画布大小
matplotlib.pyplot.plot(
    history['Test Loss'],
    label='Test Loss',
    color='blue',
    linestyle='--',
    marker='o'
)
matplotlib.pyplot.legend(loc='upper right')
matplotlib.pyplot.grid(True, linestyle=':', alpha=0.6)
matplotlib.pyplot.xlabel('Epoch', fontsize=12)
matplotlib.pyplot.ylabel('Loss', fontsize=12)
matplotlib.pyplot.title('Test Loss Curve', fontsize=14)
matplotlib.pyplot.show()

# 对测试准确率进行可视化
matplotlib.pyplot.figure(figsize=(10, 5))  # 新开一个画布
matplotlib.pyplot.plot(
    history['Test Accuracy'],
    color='red',
    label='Test Accuracy',
    linestyle='-.',
    marker='s'
)
matplotlib.pyplot.legend(loc='lower right')
matplotlib.pyplot.grid(True, linestyle=':', alpha=0.6)
matplotlib.pyplot.xlabel('Epoch', fontsize=12)
matplotlib.pyplot.ylabel('Accuracy', fontsize=12)
matplotlib.pyplot.title('Test Accuracy Curve', fontsize=14)
matplotlib.pyplot.ylim(0.0, 1.0)  # 固定Y轴范围
matplotlib.pyplot.show()
torch.save(net.state_dict(),'./lenet.pth')






