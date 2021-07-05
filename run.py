from util.model import *
from util.trainer import train, evaluate
from config import Config
from util.dataTool import RepeatDataset
import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    config = Config()

    print("Model building...")
    doubleTower = DoubleTower(symmetry=True)

    inputLayer = Passer(ifEnd=False)
    featureLayer = Passer(ifEnd=True)
    outputLayer = End()

    inputLayer.setOpt(Seq())
    featureLayer.setOpt(RNN(300, 3, 0.5))
    outputLayer.setOpt(Sig(dim=300))

    inputLayer.setSuccessor(featureLayer)
    featureLayer.setSuccessor(outputLayer)

    doubleTower.buildTower(inputLayer=inputLayer, featureLayer=featureLayer, outputLayer=outputLayer)

    doubleTower.to(config.device)
    doubleTower.train()

    print("Data loading...")
    train_data = RepeatDataset('data/label.csv', config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, drop_last=True, shuffle=True)

    test_data = RepeatDataset('data/test_label.csv', config)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, drop_last=False, shuffle=True)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(doubleTower.parameters(),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    doubleTower = train(model=doubleTower,
                        loader=train_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        config=config)

    evaluate(doubleTower, test_loader, config)