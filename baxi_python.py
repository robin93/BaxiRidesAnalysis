import pandas as pd
import numpy as np

"""data preparation for Prediction. Normalising columns and preparing variable to predict"""
print "preparing data"
raw_data = pd.read_csv("train_slave2-2.csv",header=0,sep=",")
raw_data['ridesOTP'] = raw_data['trip_count'] + raw_data['rides_cancelled'] - raw_data['rides_app']
raw_data['booked_app&OTP'] = raw_data['ridesOTP'] + raw_data['rides_app']

def LastWeekBinary(row):
	lastweekRides = row['rides_last_week']
	if lastweekRides > 0 :
		return 1
	else:
		return 0

#normalizing the trip data by row for each customer"""
raw_data['norm_trip_count'] = raw_data['trip_count']/raw_data['booked_app&OTP']
raw_data['norm_rides_app'] = raw_data['rides_app']/raw_data['booked_app&OTP']
raw_data['norm_rides_cancelled'] = raw_data['rides_cancelled']/raw_data['booked_app&OTP']
raw_data['norm_ridesOTP'] = raw_data['ridesOTP']/raw_data['booked_app&OTP']

raw_data['rides_last_week_binary'] = raw_data.apply(LastWeekBinary,axis=1)

prediction_data = raw_data[['SlaveId','age','norm_trip_count','norm_rides_app','norm_rides_cancelled','norm_ridesOTP','rides_last_week_binary']]

# print prediction_data.head(10)


prediction_data = prediction_data.iloc[np.random.permutation(len(prediction_data))]
test_data = prediction_data[42953:]
train_val_data = prediction_data[:42953]

#assess the distribution of 1 and 0 in the last week
print train_val_data['rides_last_week_binary'].value_counts()
positive_observations = train_val_data[train_val_data['rides_last_week_binary']==1]
negative_observations = train_val_data[train_val_data['rides_last_week_binary']==0]

# # print len(positive_observations)
# # print len(negative_observations)

# #since the positive observations are in excess, we undersample the negative samples to make the number equal
sampled_negative_observations = negative_observations.sample(n=len(positive_observations),random_state=10)

undersampled_data = positive_observations.append(sampled_negative_observations,ignore_index=True)
np.random.seed(100)
shuffled_undersampled_data = undersampled_data.iloc[np.random.permutation(len(undersampled_data))]

print shuffled_undersampled_data['rides_last_week_binary'].value_counts()
print shuffled_undersampled_data.head(10)


"""Prediction of last week ride using Machine learning"""

train_data_X,train_data_Y = shuffled_undersampled_data.iloc[:5000,:6],shuffled_undersampled_data.iloc[:5000,6]
val_data_X,val_data_Y = shuffled_undersampled_data.iloc[5000:,:6],shuffled_undersampled_data.iloc[5000:,6]
train_val_data_X,train_val_data_Y = shuffled_undersampled_data.iloc[:,:6],shuffled_undersampled_data.iloc[:,6]
test_data_X,test_data_Y = test_data.iloc[0:,:6],test_data.iloc[0:,6]

train_data_X.pop('SlaveId'),val_data_X.pop('SlaveId'),train_val_data_X.pop('SlaveId'),test_data_X.pop('SlaveId')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

clf = RandomForestClassifier(n_estimators=800,max_depth=5)
print "training...."
clf.fit(train_data_X,train_data_Y)
print "predicting..."
clf_predict = clf.predict(val_data_X)
print "type of validation class predict",type(clf_predict)
print "type of validation data y column",type(val_data_Y)

confusion_matrix_val = confusion_matrix(val_data_Y,clf_predict)

print "Actual - 1 , Predicted - 1   :", confusion_matrix_val[1,1]
print "Actual - 0 , Predicted - 0   :", confusion_matrix_val[0,0]
print "Actual - 1 , Predicted - 0   :", confusion_matrix_val[0,1]
print "Actual - 0 , Predicted - 1   :", confusion_matrix_val[1,0]

print "prediction hit rate",(float(confusion_matrix_val[1,1]/float(confusion_matrix_val[1,1]+confusion_matrix_val[1,0]))*100)
print "false negative     ", (float(confusion_matrix_val[0,1]/float(confusion_matrix_val[0,0]+confusion_matrix_val[0,1]))*100)

print "feature importance",clf.feature_importances_

print "training on train and val data for prediction on test data...."
clf_2 = RandomForestClassifier(n_estimators=800,max_depth=5)
clf_2.fit(train_val_data_X,train_val_data_Y)
clf_predict_test = clf_2.predict(test_data_X)

print clf_predict_test
print type(clf_predict_test)
print type(test_data_Y)

confusion_matrix_test = confusion_matrix(test_data_Y,clf_predict_test)
print "results on test data set....."
print "Actual - 1 , Predicted - 1   :", confusion_matrix_test[1,1]
print "Actual - 0 , Predicted - 0   :", confusion_matrix_test[0,0]
print "Actual - 1 , Predicted - 0   :", confusion_matrix_test[0,1]
print "Actual - 0 , Predicted - 1   :", confusion_matrix_test[1,0]
print "prediction hit rate",(float(confusion_matrix_test[1,1]/float(confusion_matrix_test[1,1]+confusion_matrix_test[1,0]))*100)
print "false negative     ", (float(confusion_matrix_test[0,1]/float(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]))*100)


#test on users with age less than 50 weeks
test_data = prediction_data[42953:]
test_data = test_data[test_data['age']<50]
young_test_data_X,young_test_data_Y = test_data.iloc[0:,:6],test_data.iloc[0:,6]
young_test_data_X.pop('SlaveId')
clf_predict_young_test = clf_2.predict(young_test_data_X)
confusion_matrix_young_test = confusion_matrix(young_test_data_Y,clf_predict_young_test)
print "results on test data set....."
print "Actual - 1 , Predicted - 1   :", confusion_matrix_young_test[1,1]
print "Actual - 0 , Predicted - 0   :", confusion_matrix_young_test[0,0]
print "Actual - 1 , Predicted - 0   :", confusion_matrix_young_test[0,1]
print "Actual - 0 , Predicted - 1   :", confusion_matrix_young_test[1,0]

print "prediction hit rate",(float(confusion_matrix_young_test[1,1]/float(confusion_matrix_young_test[1,1]+confusion_matrix_young_test[1,0]))*100)
print "false negative     ", (float(confusion_matrix_young_test[0,1]/float(confusion_matrix_young_test[0,0]+confusion_matrix_young_test[0,1]))*100)




