import numpy as np
from sklearn import datasets, tree

def load_data():
    dataset = datasets.load_iris()
    iris_data = dataset.data
    iris_label = dataset.target
    return iris_data, iris_label

def train_test_split(iris_data, iris_label):
    indices = np.random.permutation(len(iris_data))
    split_threshold = 20
    train_data = iris_data[indices[:-split_threshold]]
    train_label = iris_label[indices[:-split_threshold]]
    test_data = iris_data[indices[-split_threshold:]]
    test_label = iris_label[indices[-split_threshold:]]
    print(len(train_data))
    return train_data, train_label, test_data, test_label

def create_model():
    dt = tree.DecisionTreeClassifier(criterion='gini')
    return dt

def train_model(model, train_data, train_label):
    model.fit(train_data, train_label)
    model.score(train_data, train_label)
    return model

def test_model(model, test_data):
    predicted = model.predict(test_data)
    return predicted

def display_comparison(data, actual_label, predicted_label):
    for item, actual, predicted in zip(data, actual_label, predicted_label):
        print('{}\t{}\t{}'.format(item, actual, predicted))

def main():
    iris_data, iris_labels = load_data()
    train_data, train_label, test_data, test_label = train_test_split(iris_data, iris_labels)
    dt = create_model()
    dt = train_model(dt, train_data, train_label)
    predicted = test_model(dt, test_data)
    display_comparison(test_data, test_label, predicted)

if __name__ == '__main__':
    main()
