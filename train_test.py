import time
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from loss import model_loss
from cnn import *
from plot import *
import numpy as np


def model_struct(x_train_np, batch_size, net):
    batch_size = batch_size
    model = net(batch_size, x_train_np.shape[1], 1)
    return model, batch_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sava(state_dict, parameter_sava_location):
    return torch.save(state_dict, parameter_sava_location)


def load(parameter_load_location):
    return torch.load(parameter_load_location)


def train_CNN(x_train_np, y_train_np, epochs, lr, batch_size, net, parameter_sava_location, plot_location):
    model, batch_size = model_struct(x_train_np, batch_size, net)
    model.cuda()
    epochs = epochs
    optimizer = Adam(model.parameters(), lr=lr)

    inputs = torch.from_numpy(x_train_np).cuda().float()
    outputs = torch.from_numpy(y_train_np).cuda().float()
    tensor = TensorDataset(inputs, outputs)
    loader = DataLoader(tensor, batch_size, drop_last=True)  # shuffle= True drop_last=True

    # r2_plot = []
    mse_plot = []
    mae_plot = []

    training_starting_time = time.time()

    for epoch in range(epochs):
        avg_loss, avg_r2_score, avg_mse, pred_tensor = model_loss(model, loader, train=True, optimizer=optimizer)

        mae_plot.append(avg_loss)
        # r2_plot.append(avg_r2_score)
        mse_plot.append(avg_mse)

        print("Epoch " + str(epoch + 1) + ":\n\tMAE = " + str(avg_loss) + "\n\tMSE = " + str(
            avg_mse) + "\n\tR^2 Score = " + str(avg_r2_score))

        training_end_time = time.time() - training_starting_time
        print("\ttime: " + str(training_end_time) + "\n")

    predictions_np = np.array(pred_tensor.cpu().detach().numpy())

    state_dict = model.state_dict()
    sava(state_dict, parameter_sava_location)
    print("训练模型参数数量为: " + str(count_parameters(model)))
    plot.train_curve(mse_plot, mae_plot, plot_location)

    return predictions_np


def test_CNN(x_test_np, y_test_np, batch_size, parameter_load_location):
    model, batch_size = model_struct(x_test_np, batch_size, CNN)
    state_dict_load = load(parameter_load_location)
    model.load_state_dict(state_dict_load)
    model.eval()
    model.cuda()

    inputs = torch.from_numpy(x_test_np).cuda().float()
    outputs = torch.from_numpy(y_test_np).cuda().float()
    test_tensor = TensorDataset(inputs, outputs)
    test_loader = DataLoader(test_tensor, batch_size, drop_last=True)
    avg_loss, avg_r2_score, avg_mse, test_tensor = model_loss(model, test_loader)
    test_np = np.array(test_tensor.cpu().detach().numpy())

    print("此模型测试\n\tMAE = " + str(avg_loss) + "\n\tMSE = " + str(avg_mse) + "\n\tR^2 Score = " + str(
        avg_r2_score))

    return test_np
