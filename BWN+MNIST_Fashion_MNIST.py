import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

# 定义BWN
class BinaryLinearLayer(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinearLayer, self).__init__(*args, **kwargs)

    def forward(self, input):
        binary_weight = torch.sign(self.weight)
        return nn.functional.linear(input, binary_weight, self.bias)



# 定义模型
class BinarizedMLP(nn.Module):
    def __init__(self):
        super(BinarizedMLP, self).__init__()

        self.bwn1 = BinaryLinearLayer(784, 1024)
        self.bwn2 = BinaryLinearLayer(1024, 512)
        self.bwn3 = BinaryLinearLayer(512, 256)
        self.bwn4 = BinaryLinearLayer(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.bwn1(x)
        x = torch.relu(x)
        x = self.bwn2(x)
        x = torch.relu(x)
        x = self.bwn3(x)
        x = torch.relu(x)
        x = self.bwn4(x)

        return x

# 加载数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#MNIST数据集
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
#fashion-MNIST数据集
#train_dataset = FashionMNIST(root='./fashion_mnist_data', train=True, download=True, transform=transform)
#test_dataset = FashionMNIST(root='./fashion_mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 实例化模型、损失函数和优化器
model = BinarizedMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 存储每个 epoch 的损失和准确率
train_loss_list = []
train_accuracy_list = []
test_accuracy_list=[]

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # 将输入数据和标签移动到GPU上
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(train_loader)


    # 每个epoch中网络在训练集中验证的准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%, Testing Accuracy: {100 * test_accuracy:.2f}%')


    # 记录每个 epoch 的损失和准确率
    train_loss_list.append(average_loss)
    train_accuracy_list.append(accuracy)

# 保存训练完的模型
torch.save(model, 'BWN+MNIST+Fashion_MNIST.pth')

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()



# 测试模型
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # 将输入数据和标签移动到GPU上
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# 显示一张测试图像
def test_single_image(image):
    model.eval()
    with torch.no_grad():
        # 将输入数据移动到GPU上
        image = image.to(device)

        output = model(image)
        _, predicted = output.max(1)
        print(f'Predicted Label: {predicted.item()}')

# 从测试集中获取一张图像
test_iterator = iter(test_loader)
test_image, test_label = next(test_iterator)

# 显示图像
plt.imshow(test_image[0][0], cmap='gray')
plt.show()

# 对图像进行测试
test_single_image(test_image[0].unsqueeze(0).to(device))
