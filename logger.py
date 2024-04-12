import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 判断是否有可用的GPU，如果有就使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载Cora数据集
dataset = Planetoid(root='data/Cora', name='Cora')

# 定义一个GCN模型
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        # 第一个图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二个图卷积层
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # 全连接层
        self.fc = nn.Linear(output_dim, dataset.num_classes)
        
    def forward(self, x, edge_index):
        # 第一次图卷积，使用ReLU激活函数
        x = F.relu(self.conv1(x, edge_index))
        # 第二次图卷积
        x = self.conv2(x, edge_index))
        # Dropout操作，防止过拟合
        x = F.dropout(x, training=self.training)
        # 全连接层
        x = self.fc(x)
        # 使用log_softmax进行分类
        return F.log_softmax(x, dim=1)

# 创建GCN模型实例，并将模型移动到设备上（GPU或CPU）
model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# 定义训练函数
def train(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()

# 定义测试函数
def test(model, data):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    acc = pred[data.test_mask].eq(data.y[data.test_mask].to(device)).sum().item() / data.test_mask.sum().item()
    return acc

# 进行模型训练和测试，并输出测试集准确率
for epoch in range(200):
    train(model, optimizer, criterion, dataset[0])
    test_acc = test(model, dataset[0])
    print('Epoch: {:03d}, Test Acc: {:.4f}'.format(epoch, test_acc))
