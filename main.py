from data_processing import *
from train_test import *
from plot import *

x_o, y_o = dataset.load_csv('C:/Users/King/Desktop/Code/CNN/data/CNN_data_plus_plus.csv')
x, y, std_scale2 = dataset.std(x_o, y_o)
x_train_np, y_train_np, x_test_np, y_test_np = dataset.split(x, y, test_size=0.1)

train_test = input("请输入 train 或 test ：")
if train_test == "train":
    result_train = train_CNN(x_train_np, y_train_np, epochs=200, lr=0.001, batch_size=128, net=CNN,
                             parameter_sava_location='C:/Users/King/Desktop/Code/CNN/parameter/CNN.pt',
                             plot_location='C:/Users/King/Desktop/Code/CNN/plot/train_loss_CNN.png')
    result_train = dataset.std_inverse(result_train, std_scale2)
    origin_train = dataset.std_inverse(y_train_np, std_scale2)
    plot.train_test_plot(result_train, origin_train, plot_location1='C:/Users/King/Desktop/Code/CNN/plot/train_CNN.png',
                         plot_location2='C:/Users/King/Desktop/Code/CNN/plot/train_diff_CNN.png',
                         data_compare_location='C:/Users/King/Desktop/Code/CNN/result/train_data_compare.xlsx')

elif train_test == "test":
    result_test = test_CNN(x_test_np, y_test_np, batch_size=128,
                           parameter_load_location='C:/Users/King/Desktop/Code/CNN/parameter/CNN.pt')
    result_test = dataset.std_inverse(result_test, std_scale2)
    origin_test = dataset.std_inverse(y_test_np, std_scale2)

    plot.train_test_plot(result_test, origin_test, plot_location1='C:/Users/King/Desktop/Code/CNN/plot/test_CNN.png',
                         plot_location2='C:/Users/King/Desktop/Code/CNN/plot/test_diff_CNN.png',
                         data_compare_location='C:/Users/King/Desktop/Code/CNN/result/test_data_compare.xlsx')

else:
    print("输入错误")



