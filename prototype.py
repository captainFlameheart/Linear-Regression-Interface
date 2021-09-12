import numpy as np
import csv

def add_train_data(train_data, path):
    """ Add the train data to the specified csv file """
    row = []
    for item in train_data[0].items():
        row += item
    row.append(train_data[1])
    with open(path + ".csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def clear_train_data(path):
    """ Clear the train data in the given csv file """
    with open(path + ".csv", mode="w", newline="") as file:
        file.truncate


def pop_train_data(index, path):
    """
    Remove and return the train data with the given index in the csv file
    """
    all_train_data = get_train_data(path)
    result = all_train_data.pop(index)
    clear_train_data(path)
    for train_data in all_train_data:
        add_train_data(train_data, path)
    return result


def get_train_data(path):
    """ Get the train data from the specified csv file """
    train_data = []
    with open(path + ".csv", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            row_dict = {}
            for i in range(0, len(row) - 1, 2):
                row_dict[row[i]] = float(row[i + 1])
            train_data.append((row_dict, float(row[-1])))
    return train_data


def predict_y(x, train_data):
    """
    Return a predicted output of the input x using linear regression on the
    train data
    """
    input_identifiers = set()
    for train_data_item in train_data:
        for id in train_data_item[0]:
            input_identifiers.add(id)
    input_identifiers = list(input_identifiers)

    train_x = []
    train_y = []
    for train_data_item in train_data:
        current_train_x = []
        for id in input_identifiers:
            current_train_x.append(train_data_item[0][id] if id in train_data_item[0] else 0)
        train_x.append(current_train_x)
        train_y.append(train_data_item[1])

    weights = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
    
    test_x = []
    for id in input_identifiers:
        test_x.append(x[id] if id in x else 0)
    return np.dot(test_x, weights)


if __name__ == "__main__":
    clear_train_data("shoppinghistory")

    # 5 apples and 4 bananas cost 55 kr
    add_train_data(({"apple": 5, "banana": 4}, 55), "shoppinghistory")

    # 3 bananas cost 15 kr
    add_train_data(({"banana": 3}, 15), "shoppinghistory")

    # 2 bananas and 4 packages of milk cost 50 kr
    add_train_data(({"banana": 2, "milk": 4}, 50), "shoppinghistory")

    # What should 3 packages of milk cost?
    predicted_cost = round(predict_y({"milk": 3}, get_train_data("shoppinghistory")))
    print("3 packages of milk should cost " + str(predicted_cost) + " kr")

    # What should 1 apple cost?
    predicted_cost = round(predict_y({"apple": 1}, get_train_data("shoppinghistory")))
    print("1 apple should cost " + str(predicted_cost) + " kr")
