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
raw_data['norm_age'] = raw_data['age']/float(165)
raw_data['rides_last_week_binary'] = raw_data.apply(LastWeekBinary,axis=1)

cluster_data = raw_data[['norm_age','norm_trip_count','norm_rides_app','norm_rides_cancelled','norm_ridesOTP']]

from sklearn.cluster import KMeans

clust = KMeans(n_clusters=6,max_iter = 400)
clust.fit(cluster_data)
assignment = clust.labels_


print "inertia",clust.inertia_

print "cluster centres 6\n",clust.cluster_centers_

print "assignment", assignment

raw_data['cluster_assignment'] = assignment

print raw_data.head(10)

output_data = raw_data[['SlaveId','age','trip_count','rides_app','rides_cancelled','ridesOTP','cluster_assignment']]
output_data.to_csv('cluster_assignment_Baxi_data.csv',sep=',',index=False)


