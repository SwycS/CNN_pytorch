import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class dataset:
    def load_csv(self):
        dataset = pd.read_csv(self)
        x_o = dataset.loc[:, :'S']
        y_o = dataset.loc[:, ['yield']]

        return x_o, y_o

    def std(x_o, y_o):
        std_scale1 = preprocessing.StandardScaler().fit(x_o)
        std_scale2 = preprocessing.StandardScaler().fit(y_o)
        x = std_scale1.transform(x_o)
        y = std_scale2.transform(y_o)
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)

        return x, y, std_scale2

    def std_inverse(std, std_scale2):
        origin = std_scale2.inverse_transform(std)

        return origin

    def split(x, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)
        # shuffle = True, random_state=1
        x_train_np = x_train.to_numpy()
        y_train_np = y_train.to_numpy()

        x_test_np = x_test.to_numpy()
        y_test_np = y_test.to_numpy()

        return x_train_np, y_train_np, x_test_np, y_test_np
