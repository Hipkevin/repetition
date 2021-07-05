from tqdm import tqdm
from sklearn.metrics import classification_report


def train(model, loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):

        for idx, (q, v, Y) in enumerate(loader):
            q, v = q.to(config.device), v.to(config.device)
            Y = Y.to(config.device)

            predict = model(q, v)
            loss = criterion(predict, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")

    return model


def evaluate(model, loader, config):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (q, v, Y) in enumerate(loader):
        q, v = q.to(config.device), v.to(config.device)
        Y = Y.to(config.device)

        y_pre += model(q, v).squeeze().cpu().tolist()
        y += Y.cpu().tolist()

    y_hat = list()
    for pre in y_pre:
        if pre > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    print(classification_report(y, y_hat))