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
TEST_SURVIVED = 11

input_file_name = 'data/train.csv'
test_file_name =  'data/test.csv'

# Model Hyperparameters
epochs = 10

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# ---------------------------------------------------- * * * ---------------------------------------------------- REQUISITE FUNCTIONS

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
	if not is_number(madata[0][col]):    # convert to numbers
		encode_class = []
		for i in range(len(madata)):
			encode_class.append(madata[i][col])
		le = LabelEncoder()
		le.fit(encode_class)
		encode_class = list(le.transform(encode_class))
		for i in range(len(madata)):
			madata[i][col] = encode_class[i]

	for i in range(len(madata)):         # convert numbers to categorical values
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

def convert_np(elem):
	return np.asarray([elem])

def ready_predictions(data):
	preds = []
	for i in range(len(data)):
		preds.append(convert_np(data[i]))
	return preds

def read_file(file_name):
	raw_data = open(file_name, 'rt')
	madata = list(csv.reader(raw_data, delimiter=','))
	raw_data.close()
	return madata

# ---------------------------------------------------- * * * ---------------------------------------------------- PROCESS INPUT DATA
madata = read_file(input_file_name)
train_labels = madata[0]
madata.remove(madata[0])

t_data = delete_target_data_from_train(madata)
y_train = np.asarray(t_data)

madata = convert_to_categorical(madata, P_CLASS)
madata = convert_to_categorical(madata, SEX)
madata = convert_to_categorical(madata, EMBARKED)

input_data = get_input_data(madata, P_CLASS, SEX, AGE, FARE, EMBARKED)
x_train = np.asarray(input_data)

# ---------------------------------------------------- * * * ---------------------------------------------------- TRAIN MODEL
'''
model = Sequential()
model.add(Dense(units=64, input_dim=len(x_train), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.fit(input_data, y_train, epochs=epochs, batch_size=50)
'''
# ---------------------------------------------------- * * * ---------------------------------------------------- RETRIEVE TEST DATA - PROCESS

t_madata = read_file(test_file_name)
t_madata.remove(t_madata[0])


a = convert_to_categorical(t_madata, TEST_P_CLASS)
b = convert_to_categorical(t_madata, TEST_SEX)
c = convert_to_categorical(t_madata, TEST_EMBARKED)

test_data = get_input_data(t_madata, TEST_P_CLASS, TEST_SEX, TEST_AGE, TEST_FARE, TEST_EMBARKED)
x_test = np.asarray(test_data)

x_test = convert_to_categorical(test_data, 0)
x_test = convert_to_categorical(test_data, 1)
x_test = convert_to_categorical(test_data, 4)

x_test = np.array(x_test)

print("\n\n\n\n\n\n\n---- * * * ----")
print("\n\n\n")

model = load_model("my_model.h5")
x_test = np.asarray(x_test)

# ---------------------------------------------------- * * * ---------------------------------------------------- PREDICTIONS ON TEST DATA

raw_preds = ready_predictions(x_test)
predictions = []

test_names = get_input_data(t_madata, TEST_NAME)

for i in range(len(raw_preds)):
	predictions.append(model.predict(raw_preds[i])[0][0])

# ---------------------------------------------------- * * * ---------------------------------------------------- POST-TRAINING ANALYSIS

count = 0
nums = 0
for i in range(40):
	if predictions[i] >= 0.5:
		nums = nums + 1
		print("Prediction ", i, ": ", predictions[i], " ----------> ", test_names[i], " -- ", t_madata[i][TEST_SURVIVED])
		if (t_madata[i][TEST_SURVIVED] == '1'):
			count = count + 1

print("\n\n\nPercent: ", (count/nums)*100, "%")



# Increase test accuracy
# --- improve neural network
# Include categorical data
# --- improve convert_to_categorical
# Generative adversarial network
# --- generate a possible titanic passenger