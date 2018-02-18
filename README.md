# Titanic dataset

 - I tried to convert the non-numeric data into categorical one-hot vector data where needed
 - `model.fit()` has been unable to accept such data for now
 - So I am currently just settling with `sklearn`'s `LabelEncoder` and converting categorical data into numbers instead of one-hot versions

 TODO:
 -------
 1. Fix `x_test` reading of values
 2. Data processing: `convert_to_categorical`