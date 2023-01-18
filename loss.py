from torch.nn import L1Loss, MSELoss
from ignite.contrib.metrics.regression.r2_score import R2Score
import torch


def model_loss(model, dataset, train=False, optimizer=None):
    performance = L1Loss()
    m = MSELoss()
    score_metric = R2Score()

    avg_loss = 0
    avg_mse = 0
    avg_score = 0
    count = 0
    pred = []

    for input, output in iter(dataset):
        predictions = model.forward(input)
        loss = performance(predictions, output)
        mse = m(predictions, output)
        score_metric.update([predictions, output])
        score = score_metric.compute()
        pred.append(predictions)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss += loss.item()
        avg_mse += mse.item()
        avg_score += score
        count += 1

    pred_tensor = torch.cat(pred, dim=0)

    return avg_loss / count, avg_score / count, avg_mse / count, pred_tensor
