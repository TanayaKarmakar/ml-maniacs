from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

cols = ['hr','season', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

train_data = 'dataset/Bike-Sharing-Dataset/hour.csv'
#test_data = 'dataset/Bike-Sharing-Dataset/day.csv'

def load_data():
    global cols
    dataset = pd.read_csv(train_data, usecols = cols)
    print(dataset.head())
    return dataset

def visualize_data(dataset):
    variables = cols[1:-3]
    for index, variable in enumerate(variables):
        plt_num = index + 1
        data = dataset[variable]
        plt.subplot(5,2,plt_num)
        plt.hist(data)
        plt.title('{} : {}'.format(variable, 'cnt'))

    plt.subplots_adjust(wspace = 0.3, hspace = 0.8)
    plt.show()

def perform_multivariate_analysis(dataset):
    variables = cols[:-3]
    n_row = 5
    n_col = 2
    fig, axes = plt.subplots(n_row, n_col)
    row_col_dict_list = create_row_col_list(n_row, n_col)
    for index, variable in enumerate(variables):
        data = dataset[[variable, 'cnt']]
        item = row_col_dict_list[index]
        print(str(item['row'])+ " : " + str(item['col']))
        axes[item['row'], item['col']].boxplot(data)


    fig.subplots_adjust(wspace=0.3, hspace=0.8)
    plt.show()

def create_row_col_list(n_row, n_col):
    row_col_dict_list = []
    for row in range(n_row):
        for col in range(n_col):
            row_col_dict_list.append({'row': row, 'col': col})

    return row_col_dict_list

def perform_feature_engg(dataset):
    pass

def split_train_test(dataset):
    train_data, test_data, train_label, test_label = train_test_split(dataset.iloc[:,0:-3],
                                                                      dataset.iloc[:,-1],
                                                                      test_size=0.33,
                                                                      random_state=42)
    train_data.reset_index(inplace=True)
    train_label = train_label.reset_index()
    test_data.reset_index(inplace=True)
    test_label = test_label.reset_index()

    return train_data, train_label, test_data, test_label

def build_decisiontree_regressor_model():
    model = DecisionTreeRegressor()
    return model

def train_model(model, train_data, train_label):
    print('training model')
    model.fit(train_data, train_label)
    model.score(train_data, train_label)

def test_model(model, test_data):
    print('model prediction')
    prediction = model.predict(test_data)
    return prediction

def perform_comparison(model, prediction, test_data, test_label):
    # for data, pred, actual in zip(test_data, prediction, test_label):
    #     print('{}\t{}\t{}'.format(data, pred, actual))
    prediction_score = model.score(test_data, test_label)
    mean_sq_error = mean_squared_error(test_label, prediction)
    print("Score :{:.3f}".format(prediction_score))
    print('MSE: %.2f'% mean_sq_error)

def main():
    dataset = load_data()
    visualize_data(dataset)
    #perform_multivariate_analysis(dataset) need to solve the issue
    model = build_decisiontree_regressor_model()
    train_data, train_label, test_data, test_label = split_train_test(dataset)
    train_model(model, train_data, train_label)
    prediction = test_model(model, test_data)
    perform_comparison(model, prediction, test_data, test_label)


main()


