# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
import numpy as np

import csv

SURVIVED = 1

PASSENGER_ID = 0
P_CLASS = 1
NAME = 2
SEX = 3
AGE = 4
SIB_SP = 5
PARCH = 6
TICKET = 7
FARE = 8
CABIN = 9
EMBARKED = 10

TEST_PASSENGER_ID = 0
TEST_P_CLASS = 1
TEST_NAME = 2
TEST_SEX = 3
TEST_AGE = 4
TEST_SIB_SP = 5
TEST_PARCH = 6
TEST_TICKET = 7
TEST_FARE = 8
TEST_CABIN = 9
TEST_EMBARKED = 10

input_file_name = 'data/train.csv'
test_file_name =  'data/test.csv'

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

def delete_target_data_from_train(madata):
	target = []
	for i in range(len(madata)):
		target.append(madata[i][SURVIVED])
		del madata[i][SURVIVED]
	return target

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass

	return False

def convert_to_categorical(madata, col):
	cat_class = []
	if not is_number(madata[0][col]):
		encode_class = []
		for i in range(len(madata)):
			encode_class.append(madata[i][col])
		le = LabelEncoder()
		le.fit(encode_class)
		encode_class = list(le.transform(encode_class))
		for i in range(len(madata)):
			madata[i][col] = encode_class[i]

	for i in range(len(madata)):
		cat_class.append(madata[i][col])
	# cat_class = keras.utils.np_utils.to_categorical(cat_class)
	# cat_class = cat_class.astype(int)
	for i in range(len(madata)):
		madata[i][col] = cat_class[i]
		#madata[i][col] = list(cat_class[i])  # -----
	return madata

def get_input_data(madata, *cols):
	input_data = []
	columns = list(cols)
	for i in range(len(madata)):
		input_data.append([])
		for col in columns:
			input_data[i].append(madata[i][col])
	return input_data

# ----------------------------------------------------

input_size = 1
epochs = 300

raw_data = open(input_file_name, 'rt')
madata = list(csv.reader(raw_data, delimiter=','))
train_labels = madata[0]
madata.remove(madata[0])
raw_data.close()

t_data = delete_target_data_from_train(madata)
y_train = np.asarray(t_data)

# Process data
madata = convert_to_categorical(madata, P_CLASS)
madata = convert_to_categorical(madata, SEX)
madata = convert_to_categorical(madata, EMBARKED)

input_data = get_input_data(madata, P_CLASS)
x_train = np.asarray(input_data)

'''
for i in range(len(input_data)):
	print(input_data[i])
'''




# Create model
model = Sequential()
model.add(Dense(units=64, input_dim=input_size, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(input_data, y_train, epochs=epochs, batch_size=50)


raw_data = open(test_file_name, 'rt')
t_madata = list(csv.reader(raw_data, delimiter=','))
t_madata.remove(t_madata[0])
raw_data.close()

print("Data 0: ", t_madata[0], "\n\n\n")
'''
a = convert_to_categorical(t_madata, TEST_P_CLASS)
b = convert_to_categorical(t_madata, TEST_SEX)
c = convert_to_categorical(t_madata, TEST_EMBARKED)

print(a[0])
print("\n\n\n")
print(b[0])
print("\n\n\n")
print(c[0])
'''
test_data = get_input_data(t_madata, TEST_P_CLASS)
x_test = np.asarray(test_data)

print("\n\n\n\n\n\n\n---- * * * ----")

print("\n\n\n")
print(x_train.shape)
# model.predict(np.asarray([x_train[0]]))
model.predict(x_test[0])


'''
predictions = model.predict(x_test)
pred_classes = model.predict_classes(x_test)

raw_data = open(test_file_name, 'rt')
x_test = np.loadtxt(raw_data, delimiter=',', dtype='str', skiprows=1, usecols=(TEST_NAME, TEST_NAME+1))
raw_data.close()
raw_data = open(test_file_name, 'rt')
survivor = np.loadtxt(raw_data, delimiter=',', skiprows=1, usecols=(TEST_AGE+7))
raw_data.close()

for i in range(40):
	if predictions[i] < 0.5:
		print("Prediction ", i, ": ", predictions[i], " ----------> ", x_test[i], "    ", survivor[i])
	else:
		print("Prediction ", i, ": ", predictions[i])

print("\n\n\n\n\n\n\n\n\n")

for i in range(40):
	if predictions[i] > 0.5:
		print("Prediction ", i, ": ", predictions[i], " ----------> ", x_test[i], "    ", survivor[i])
	else:
		print("Prediction ", i, ": ", predictions[i])


'''

# Load data succesfully
# --- Numpy load, formatting
# Train data to produce desired output: dead or alive
# --- Declare model, add layers, compile, fit data
# Test model on test data
# --- Evaluate on test data