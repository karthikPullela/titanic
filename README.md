# Titanic dataset

 - I tried to convert the non-numeric data into categorical one-hot vector data where needed
 - `model.fit()` has been unable to accept such data for now
 - So I am currently just settling with `sklearn`'s `LabelEncoder` and converting categorical data into numbers instead of one-hot versions

 TODO:
 -------
1. Increase test accuracy --> improve model
2. Data processing: `convert_to_categorical`
3. Create a passenger who would likely survive or die on Titanic
4. Maybe, based on user characteristics, predict if user would survive the titanic
