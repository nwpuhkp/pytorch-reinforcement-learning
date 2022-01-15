#三个输入
#epsilons - 要用于运行的epsilon值的列表。在列表中保留0是很重要的，因为它代表了原始测试集上的模型性能。
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

epsilons = [0,.05,.1,.15,.2,.25,.3]#epsilon越大，扰动越明显，但在降低模型精度方面攻击越有效。
#pretrained_model - 表示使用 pytorch/examples/mnist进行训练的预训练MNIST模型的路径
pretrained_model = "data/lenet_mnist_model.pth"
#use_cuda - 如果需要和可用，使用CUDA的布尔标志。
use_cuda = True
#受攻模型
#定义模型和加载数据，初始化模型并加载预先训练的权重
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,10,kernel_size=5)
#         self.conv2 = nn.Conv2d(10,20,kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320,50)
#         self.fc2 = nn.Linear(50,10)
#
#     def forward(self,x):
#         x = F.relu(F.max_pool2d(self.conv1(x),2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
#         x = x.view(-1,320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x,training = self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x,dim=1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


#测试数据集和数据加载器
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',train = False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1,shuffle=True)

# print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = Net().to(device)

model.load_state_dict(torch.load(pretrained_model,map_location='cpu'))

model.eval()

#定义一个通过打乱原始输入来生成对抗性示例的函数
def fgsm_attack(image,epsilon,data_grad):#image原始图像，epsilon是像素级干扰量，data_grad是关于输入图像的损失
    sign_data_grad = data_grad.sign()#收集数据梯度的元素级符号
    perturbed_image = image + epsilon * sign_data_grad#通过调整输入图像的每个像素来创建受扰动的图像
    perturbed_image = torch.clamp(perturbed_image,0,1)#为了保持数据的原始范围，将扰动后的图像截取范围在 [0,1]。
    return  perturbed_image

#测试函数，每次调用都在MNIST测试集上执行一个完整的测试步骤，然后给出一个最终准确性报告
#对于测试集中的每个样本，该函数计算和输入数据data_grad相关的损失梯度，用fgsm_attack perturbed_data创建一个干扰图像，然后检查干扰的例子是否是对抗性的。
def test(model,device,test_loader,epsilon):
    correct = 0#精度
    adv_examples = []

    #循环测试集中的所有示例
    for data,target in test_loader:
        data,target = data.to(device),target.to(device)#gpu训练
        #设置张量的 requires_grad 属性。  对攻击很重要
        data.requires_grad = True
        output = model(data)#向前传递
        init_pred = output.max(1,keepdim = True)[1]#获取最大对数概率的索引
        #如果最初的预测是错误的，不要打扰攻击，继续前进
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output,target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data#收集数据梯度
        #调用FGSM
        perturbed_data = fgsm_attack(data,epsilon,data_grad)
        #重新分类扰动图像
        output = model(perturbed_data)
        #检查是否成功
        final_pred = output.max(1,keepdim = True)[1]
        if final_pred.item() == target.item():
            correct += 1
            #保存epsilon=0示例的特殊情况
            if(epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(),final_pred.item(),adv_ex))

        else:
            #保存一些例子便于后面可视化
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct/float(len(test_loader))
    print("Epsilon:{}\tTest Accuracy = {} / {} = {}".format(epsilon,correct,len(test_loader),final_acc))

    return  final_acc,adv_examples

#运行攻击操作
#其中epsilon=0用例表示原始未受攻击的测试准确性。
accuracies = []
examples = []

for eps in epsilons:
    acc,ex = test(model,device,test_loader,eps)
    accuracies.append(acc)
    examples.append(ex)
    #随着epsilon的增加，预期测试的准确性降低了。

#epsilon虽然值是线性间隔的，但是降低的曲线趋势却并非线性
# plt.figure(figsize=(5,5))
# plt.plot(epsilons,accuracies,"*-")
# plt.yticks(np.arange(0,1.1,step = 0.1))
# plt.xticks(np.arange(0, .35, step=0.05))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()

#随着epsilon 的增加，测试精度降低，但扰动变得更容易察觉。必须在这之间进行权衡。
#画出例子
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
