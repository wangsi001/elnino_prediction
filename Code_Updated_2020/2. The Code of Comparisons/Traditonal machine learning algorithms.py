# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:44:17 2019

@author: wangci
"""

# direct multi-step forecast by lead time
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt
import numpy as np

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[0:816], data[816::]
	# restructure into windows of weekly data
	print ("11111111111")
	print (train.shape)
	print (test.shape)
	train = array(split(train, len(train)/12))
	print (train)
	test = array(split(test, len(test)/12))
	print (test)
	return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(name,actual, predicted):
	scores = list()
	#npnum=np.random.randint(1,1000000,1)
	# calculate an RMSE score for each day
	plt.figure(figsize=(20,8))
	index_start = np.array(range(1,13))
	plt.title("Actual Values And Predicted Values",fontstyle="italic",fontsize=24)
	plt.plot(index_start,actual.reshape(-1),marker=".",label='Actual Values')
	plt.plot(index_start,predicted.reshape(-1),color="red",marker="o",label='Predicted Values')

	plt.xlabel('Time_Step(Month)',fontsize=14)
	plt.ylabel('values(℃)',fontsize=14)
	plt.legend(loc=4,fontsize=14) 
	plt.savefig('Nomalized_12_months_ConvLSTM_Encoder-Decoder Model/{0}.png'.format(name))
	plt.show()      
	
	for i in range(actual.shape[1]):
		# calculate mse
		#print ("--------------")
		#print (actual[:, i])
		#print (predicted[:, i])
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# prepare a list of ml models
def get_models(models=dict()):
    
    # non-linear models
	print ("non-linear_models:")
	models['svmr'] = SVR()    
	models['knn'] = KNeighborsRegressor(n_neighbors=12)
	'''models['cart'] = DecisionTreeRegressor()
	models['extra'] = ExtraTreeRegressor()
    models['svr_lin']=SVR(kernel='linear', C=1e3)
	models['svr_poly']=SVR(kernel='poly', C=1e3, degree=3)
	models['svr_rbf']=SVR(kernel='rbf', C=1e3, gamma=0.1)'''
	
	#ensemble models
	print ("ensemble_models:")
	n_trees = 100
	models['ada'] = AdaBoostRegressor(n_estimators=n_trees)

	'''
	models['bag'] = BaggingRegressor(n_estimators=n_trees)    
    models['rf'] = RandomForestRegressor(n_estimators=n_trees)
	models['et'] = ExtraTreesRegressor(n_estimators=n_trees)
	models['gbm'] = GradientBoostingRegressor(n_estimators=n_trees)'''

    #linear models
	print ("linear_models:")
	models['lr'] = LinearRegression()

	models['ridge'] = Ridge()
	'''models['en'] = ElasticNet()
	models['lasso'] = Lasso()
	models['huber'] = HuberRegressor()
	models['lars'] = Lars()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['ranscac'] = RANSACRegressor()
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))'''
	return models

# create a feature preparation pipeline for a model
def make_pipeline(model):
	steps = list()
	# standardization
	steps.append(('standardize', StandardScaler()))
	# normalization
	steps.append(('normalize', MinMaxScaler()))
	# the model
	steps.append(('model', model))
	# create pipeline
	pipeline = Pipeline(steps=steps)
	return pipeline

# # convert windows of weekly multivariate data into a series of total power
def to_series(data):
	# extract just the total power from each week
	series = [week[:, 2] for week in data]
	# flatten into a single series
	series = array(series).flatten()
	return series

# convert history into inputs and outputs
def to_supervised(history, n_input, output_ix):
	# convert history to a univariate series
	data = to_series(history)

	X, y = list(), list()
	ix_start = 0
 
	ix_end = ix_start + n_input
	ix_output = ix_end + output_ix

	# step over the entire history one time step at a time
	for i in range(len(data)):
		# define the end of the input sequence
		ix_end = ix_start + n_input
		ix_output = ix_end + output_ix
     
		# ensure we have enough data for this instance
		if ix_output < len(data):
			X.append(data[ix_start:ix_end])
			y.append(data[ix_output])
		# move along one time step
		ix_start += 1
	return array(X), array(y)

# fit a model and make a forecast
def sklearn_predict(model, history, n_input):
	yhat_sequence = list()
	# fit a model for each forecast day
	for i in range(12):
		# prepare data
		train_x, train_y = to_supervised(history, n_input, i)
		# make pipeline
		pipeline = make_pipeline(model)
		# fit the model
		pipeline.fit(train_x, train_y)
		# forecast
		x_input = array(train_x[-1, :]).reshape(1,n_input)
		yhat = pipeline.predict(x_input)[0]
		# store
		yhat_sequence.append(yhat)
	return yhat_sequence

# evaluate a single model
def evaluate_model(name,model, train, test, n_input):
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = sklearn_predict(model, history, n_input)
		# store the predictions
		#print (yhat_sequence)
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	predictions = array(predictions)
	print (predictions)
	# evaluate predictions days for each week
	score, scores = evaluate_forecasts(name,test[:, :, 2], predictions)
	return score, scores,predictions

# load the new file
#dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
dataset = read_csv('ONI_whole_data.txt', header=0)
print (dataset.head())
print (dataset.shape)

# split into train and test
train, test = split_dataset(dataset.values)
# prepare the models to evaluate
models = get_models()
n_input = 12
# evaluate each model
days = ['1', '2', '3', '4', '5', '6', '7','8', '9', '10', '11', '12']
prediction_name={}
for name, model in models.items():
	# evaluate and get scores
	score, scores,predictions= evaluate_model(name,model, train, test, n_input)
	# summarize scores
	summarize_scores(name, score, scores)
	# plot scores
	pyplot.plot(days, scores, marker='o', label=name)
	prediction_name[name]=predictions
# show plot
pyplot.legend()
pyplot.show()

for i in prediction_name:
   print (i)
   print (prediction_name[i])