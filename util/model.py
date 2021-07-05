import torch
import torch.nn as nn

#################
# 使用建造者模式以及职责链模式完成自动标重模型的解耦
# Tower和DoubleTower均属于建造者
# Algo为职责链的父类，通过nn.Module的forward函数特性，完成handle函数的任务
# 职责链为带指针的单向链表，只需要在顶端发起一次执行即可
# successor_n-1 --> successor_n
#################


class Algo(nn.Module):
    def __init__(self):
        super(Algo, self).__init__()
        self.successor = None

    # 设置指针
    def setSuccessor(self, successor):
        self.successor = successor


class Passer(Algo):
    """
    职责链（单向链表）节点类
    通过ifEnd变量进行尾指针判别
    opt必须实现forward或者__call__
    """
    def __init__(self, ifEnd=False):
        super(Passer, self).__init__()
        self.__opt = None
        self.ifEnd = ifEnd

    def setOpt(self, opt):
        self.__opt = opt

    def forward(self, x):

        assert self.__opt is not None, "Please set opt"
        assert hasattr(self.__opt, "forward"), "Please implement forward in opt"
        assert self.successor is not None, "Please set successor"

        x = self.__opt(x)

        if self.ifEnd:
            return x
        else:
            return self.successor(x)


class End(nn.Module):
    """
    职责链终止
    由于标重模型末端输入包含两个向量，故新建
    """
    def __init__(self):
        super(End, self).__init__()
        self.__opt = None

    def setOpt(self, opt):
        self.__opt = opt

    def forward(self, v1, v2):
        assert self.__opt is not None, "Please set opt"

        return self.__opt(v1, v2)


class Avg(nn.Module):
    def __init__(self):
        super(Avg, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=1) / x.size(0)


class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()

    def forward(self, x):
        return x


class NoneFeature(nn.Module):
    def __init__(self):
        super(NoneFeature, self).__init__()

    def forward(self, x):
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_layer_num, dropout):
        super(RNN, self).__init__()

        self.lstm = self.lstm = nn.LSTM(input_size=input_size,
                                        hidden_size=input_size // 2,
                                        num_layers=hidden_layer_num,
                                        dropout=dropout,
                                        bidirectional=True,  # 设置双向
                                        batch_first=True)  # batch为张量的第一个维度

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.sum(x, dim=1)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn(x).squeeze()
        x = x.flatten(1)

        return x


class Cosine(nn.Module):
    def __init__(self):
        super(Cosine, self).__init__()

    def forward(self, v1, v2):
        return torch.cosine_similarity(v1, v2, dim=1)


class Sig(nn.Module):
    def __init__(self, dim):
        super(Sig, self).__init__()
        self.fc = nn.Linear(dim * 2, 1)

    def forward(self, v1, v2):
        x = torch.cat((v1, v2), dim=1)
        x = self.fc(x).squeeze()
        return torch.sigmoid(x)


class Tower(nn.Module):
    def __init__(self):
        super(Tower, self).__init__()

        self.inputLayer = None
        self.featureLayer = None

    def buildTower(self, inputLayer, featureLayer):
        self.inputLayer = inputLayer
        self.featureLayer = featureLayer

    def forward(self, x):
        if self.inputLayer:
            x = self.inputLayer(x)

        """ 在头节点发起一次执行即可 """
        # if self.featureLayer:
        #     x = self.featureLayer(x)

        return x


class DoubleTower(nn.Module):
    def __init__(self, symmetry=True):
        super(DoubleTower, self).__init__()

        self.outputLayer = None
        self.single = None

        if symmetry:
            self.single = Tower()
        else:
            self.left = Tower()
            self.right = Tower()

    def buildTower(self, inputLayer, featureLayer, outputLayer):
        self.outputLayer = outputLayer

        if self.single:
            self.single.buildTower(inputLayer, featureLayer)
        else:
            self.left.buildTower(inputLayer, featureLayer)
            self.right.buildTower(inputLayer, featureLayer)

    def forward(self, q1, q2):
        assert self.outputLayer is not None, "Please setup output layer."

        if self.single:
            v1 = self.single(q1)
            v2 = self.single(q2)
        else:
            v1 = self.left(q1)
            v2 = self.right(q2)

        return self.outputLayer(v1, v2)
