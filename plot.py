import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd


class plot:
    def train_curve(mseplot, maeplot, plot_location):
        plt.plot(mseplot, lw=1, c='red', label='MSE')
        plt.plot(maeplot, lw=1, c='b', label='MAE')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_location)
        plt.show()

    def train_test_plot(result_train, origin_train, plot_location1, plot_location2, data_compare_location):
        plt.figure(figsize=(50, 30))
        plt.plot(result_train, lw=1, c='red', label='F')
        plt.plot(origin_train, lw=1, c='b', label='T')

        plt.xlabel('Total')
        plt.ylabel('data')
        plt.legend()
        plt.savefig(plot_location1)
        plt.show()

        train_test_mae = mean_absolute_error(origin_train[:result_train.shape[0], ], result_train)
        if result_train.all == result_train.all:
            print('Train_MAE: %.2f ' % train_test_mae)

        else:
            print('Test_MAE: %.2f ' % train_test_mae)

        result_pd = pd.DataFrame(result_train)
        origin_pd = pd.DataFrame(origin_train[:result_train.shape[0], ])

        total = pd.concat([result_pd, origin_pd], axis=1)
        total.columns = ('predict', 'origin')
        diff = total["predict"] - total["origin"]
        diff = diff.to_frame("diff")
        diff_rate = (total["predict"] - total["origin"]) / total["origin"] * 100
        diff_rate = diff_rate.to_frame("diff_rate %")
        total = pd.concat([total, diff], axis=1)
        total = pd.concat([total, diff_rate], axis=1)
        total.to_excel(data_compare_location)

        plt.plot(diff_rate, lw=1, c='b', label='F')
        plt.xlabel('Total')
        plt.ylabel('diff_rate')
        plt.legend()
        plt.savefig(plot_location2)
        plt.show()
