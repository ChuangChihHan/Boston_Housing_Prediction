"""
File: boston_housing_competition.py
Name: Maggie
--------------------------------
This file demonstrates how to analyze boston housing dataset.
Start with data preprocessingusing pandas, numpy and scipy, and then applying
sklearn libraries to build models and evaluate models' performance using mean square errors.
"""

import pandas as pd
from scipy import stats
import numpy as np
from sklearn import preprocessing, linear_model, tree, ensemble, model_selection, metrics

# file name
TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

# catch test data's missing data
nan_cache = {}


def main():
	# data preprocessing
	train_data, y_train, val_data, y_val = data_preprocessing(TRAIN_FILE, mode='train')
	# standardization
	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)
	val_data = standardizer.transform(val_data)

	# test data
	test_data = data_preprocessing(TEST_FILE, mode='test')
	test_data = standardizer.transform(test_data)

	# model building
	#############################################
	# linear regression
	h = linear_model.LinearRegression()
	classifier = h.fit(train_data, y_train)
	acc = classifier.score(train_data, y_train)
	train_predict = classifier.predict(train_data)
	l_predictions = classifier.predict(val_data)
	linear_prediction = classifier.predict(test_data)
	print('Linear Regression Training Accuracy:', acc)
	print(f'training RMS Error: {metrics.mean_squared_error(train_predict, y_train) ** 0.5}')
	print(f'validation RMS Error: {metrics.mean_squared_error(l_predictions, y_val) ** 0.5}')
	# print(f'Linear Regression prediction: {l_predictions}')

	#############################################
	# decision tree
	d_tree = tree.DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
	d_tree.fit(train_data, y_train)
	acc = d_tree.score(train_data, y_train)
	d_train_predict = d_tree.predict(train_data)
	d_predictions = d_tree.predict(val_data)
	decision_tree_prediction = d_tree.predict(test_data)
	print('Decision Tree Training Accuracy:', acc)
	print(f'training RMS Error: {metrics.mean_squared_error(d_train_predict, y_train)**0.5}')
	print(f'validation RMS Error: {metrics.mean_squared_error(d_predictions, y_val)**0.5}')

	# print(f'Decision Tree prediction: {d_predictions}')
	tree.export_graphviz(d_tree, out_file='tree_graph')

	#############################################
	# random forest
	forest = ensemble.RandomForestRegressor(max_depth=5, min_samples_leaf=5)
	forest.fit(train_data, y_train)
	acc = forest.score(train_data, y_train)
	r_train_predict = forest.predict(train_data)
	r_predictions = forest.predict(val_data)
	random_forest_prediction = forest.predict(test_data)
	print('Random Forest Training Accuracy:', acc)
	print(f'training RMS Error: {metrics.mean_squared_error(r_train_predict, y_train)**0.5}')
	print(f'validation RMS Error: {metrics.mean_squared_error(r_predictions, y_val)**0.5}')

	# outfile
	out_file('linear_regreesion_prediction_', linear_prediction)
	out_file('decision_tree_prediction_', decision_tree_prediction)
	out_file('random_forest_prediction_', random_forest_prediction)


def data_preprocessing(filename, mode='train'):
	"""
	@param filename: name of the file you want to preprocess its data
	@param mode: default is train
	@return: if mode = 'train' then return training data and true label,
	if mode = 'test' then return testing data
	"""
	data = pd.read_csv(filename)
	if mode == 'train':
		train_data, val_data = model_selection.train_test_split(data, test_size=0.5)
		features = ['medv', 'crim', 'indus', 'rm', 'age', 'dis', 'rad', 'tax', 'black', 'lstat']
		# use average/ median to fill in missing data
		# drop missing data
		train_data = train_data[features].dropna()
		val_data = val_data[features].dropna()

		# crime
		crim_mean = round(data['crim'].mean(), 3)
		train_data['crim'].fillna(crim_mean, inplace=True)
		val_data['crim'].fillna(crim_mean, inplace=True)

		# indus
		indus_mean = round(data['indus'].mean(), 3)
		train_data['indus'].fillna(indus_mean, inplace=True)
		val_data['indus'].fillna(indus_mean, inplace=True)

		# rm
		rm_mean = round(data['rm'].mean(), 3)
		train_data['rm'].fillna(rm_mean, inplace=True)
		val_data['rm'].fillna(rm_mean, inplace=True)

		# age
		age_mean = round(data['age'].mean(), 3)
		train_data['age'].fillna(age_mean, inplace=True)
		val_data['age'].fillna(age_mean, inplace=True)

		# dis
		dis_mean = round(data['dis'].mean(), 3)
		train_data['dis'].fillna(dis_mean, inplace=True)
		val_data['dis'].fillna(dis_mean, inplace=True)

		# rad
		rad_median = round(data['rad'].median(), 3)
		train_data['rad'].fillna(rad_median, inplace=True)
		val_data['rad'].fillna(rad_median, inplace=True)

		# tax
		tax_median = round(data['tax'].median(), 3)
		train_data['tax'].fillna(tax_median, inplace=True)
		val_data['tax'].fillna(tax_median, inplace=True)

		# black
		black_mean = round(data['black'].mean(), 3)
		train_data['black'].fillna(black_mean, inplace=True)
		val_data['black'].fillna(black_mean, inplace=True)

		# lstat
		lstat_mean = round(data['lstat'].mean(), 3)
		train_data['lstat'].fillna(lstat_mean, inplace=True)
		val_data['lstat'].fillna(lstat_mean, inplace=True)

		# drop outliers
		train_data = train_data[(np.abs(stats.zscore(train_data)) < 3).all(axis=1)]
		val_data = val_data[(np.abs(stats.zscore(val_data)) < 3).all(axis=1)]

		# y variable
		y_train = train_data['medv']
		y_val = val_data['medv']

		# x variable
		features = ['crim', 'indus', 'rm', 'age', 'dis', 'rad', 'tax', 'black', 'lstat']
		train_data = train_data[features]
		val_data = val_data[features]

		return train_data, y_train, val_data, y_val
	elif mode == 'test':
		features = ['crim', 'indus', 'rm', 'age', 'dis', 'rad', 'tax', 'black', 'lstat']
		data = data[features].dropna()
		data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

		return data


def out_file(filename, predictions):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID, medv\n')
		start_id = 3
		for ans in predictions:
			out.write(str(start_id) + ',' + str(ans) + '\n')
			start_id += 1
	print('===============================================')


if __name__ == '__main__':
	main()
