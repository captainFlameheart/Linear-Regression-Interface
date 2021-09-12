# Linear Regression Interface

Final project for the Building AI course

## Summary

An interface for storing input and output pairs in .csv files and for predicting the output of an input using
linear regression on the input and output pairs in a .csv file. This could for example be used to predict
the cost of a shopping list given previous shopping lists and their costs.

## How is it used?

The interface is in the form of a python module with functions for:
* Adding train data from a .csv file
* Removing train data from a .csv file
* Reading the train data of a .csv file
* Predicting the output of an input by using linear regression on the train data in a .csv file

Code example:
```
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
```

## Data sources and AI methods

The user determines how the train data is collected.
The AI method used is linear regression which is a supervised Machine Learning model.

## Acknowledgments

Copyright (c) 2005-2021, NumPy Developers
