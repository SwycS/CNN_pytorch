import torch
from torch.utils.data import DataLoader, TensorDataset
from cnn import *


def test_CNN(x_test_np, y_test_np):
    model.eval()

    inputs = torch.from_numpy(x_test_np).cuda().float()  # !!!!!!!!
    outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).cuda().float()  # !!!!!!!!.cuda()

    test_tensor = TensorDataset(inputs, outputs)
    test_loader = DataLoader(test_tensor, batch_size, drop_last=True)
    # shuffle=True,
    testing_startingTime = time.time()

    avg_loss, avg_r2_score, avg_MSE = model_loss(model, test_loader)

    print("The model's L1 loss is " + str(avg_loss) + "\n\tR^2 Score = " + str(avg_r2_score) + "\n\tMSE = " + str(
        avg_MSE))
    testing_endTime = time.time() - testing_startingTime
    print(testing_endTime)

    x_two = x_t.reshape(-1, 1)
    x_two_df = pd.DataFrame(x_two)
    y_test_f_df = pd.DataFrame(y_test_f[:x_two.shape[0], ])
    # y_test_f_df = pd.DataFrame(y_test_f[:1280,])
    Total = pd.concat([x_two_df, y_test_f_df], axis=1)
    Total.to_excel('C:/Users/King/Desktop/Code/CNN/parameter/distinction.xlsx')
