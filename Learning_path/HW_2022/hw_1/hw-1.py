import math
import numpy as np
import pandas as pd
import os
import csv
# from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# 保证实验可重复性，设置随机种子


def same_seed(seed):
    torch.backends.cudnn.deterministic = True  # 保证每次卷积操作结果相同，增加可重复性
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU上的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

# 划分数据集


def train_valid_split(data_set, valid_ratio, seed):
    # data_set训练集，valid_ratio验证比例，作为验证集
    valid_data_size = int(len(data_set) * valid_ratio)  # 验证集的大小
    train_data_size = len(data_set) - valid_data_size  # 训练集的大小
    train_data, valid_data = random_split(data_set, [
                                          train_data_size, valid_data_size], generator=torch.Generator().manual_seed(seed))
    return list(train_data), list(valid_data)

# 选择特征


def select_feat(train_data, valid_data, test_data, select_all=True):
    train_data = np.array([data for data in train_data])
    valid_data = np.array([data for data in valid_data])

    y_train = train_data[1:, -1]
    y_valid = valid_data[1:, -1]

    raw_x_train = train_data[1:, 1:-1]
    raw_x_valid = valid_data[1:, 1:-1]
    raw_x_test = test_data[1:, 1:]

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

# 数据集


class COVID19Dataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        if targets is None:  # 做预测，测试集，只用 features
            self.targets = targets  # targets 即 lables
        else:  # 做训练，训练集
            self.targets = torch.FloatTensor(targets)

    def __getitem__(self, idx):
        if self.targets is None:  # 测试集
            return self.features[idx]
        else:  # 训练集
            return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

# 神经网络


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'seed': 1314520,
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epochs': 2000,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,
    'save_path': './models/model.ckpt'
}

# 训练过程


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        # train_pbar = tqdm(train_loader, position=0, leave=True)
        # 加载进度条
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
            # 显示训练过程
            # train_pbar.set_description(f'Epoch[{epoch + 1}/{n_epochs}]')
            # train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # valid loop
        model.eval()
        loss_record = []

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(
            f'Epoch[{epoch + 1}/{n_epochs}]: Train loss:{mean_train_loss:.4f},Valid loss:{mean_valid_loss:.4f}')
        writer.add_scalar('Loss/Valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\n Model is not improving, so we halt the training session.')
            return


# 设置随机种子
same_seed(config['seed'])
# 读取数据
train_data = pd.read_csv(os.path.join(
    'Learning_path', 'HW_2022', 'hw_1', 'covid.train_new.csv')).values
test_data = pd.read_csv(os.path.join(
    'Learning_path', 'HW_2022', 'hw_1', 'covid.test_un.csv')).values

# 划分数据集
train_data, valid_data = train_valid_split(
    train_data, config['valid_ratio'], config['seed'])
print(
    f"train_data size : {len(train_data)}, valid_data size : {len(valid_data)}, test_data size : {test_data.shape[0]}")

# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feat(
    train_data, valid_data, test_data, config['select_all'])
print(f'the number of features : {x_train.shape[1]}')
# 构造数据集
train_dataset = COVID19Dataset(x_train, y_train)
valid_dataset = COVID19Dataset(x_valid, y_valid)
test_dataset = COVID19Dataset(x_test)
# 准备Dataloader
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(
    valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# 开始训练
model = My_Model(input_dim=x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)
