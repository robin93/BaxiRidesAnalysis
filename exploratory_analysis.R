raw_data = read.table("train_slave2-2.csv",header = TRUE,sep = ",")
raw_data

head(raw_data)
plot(raw_data$trip_count,raw_data$age,xlim = range(1,100),ylim = range(0,180))
plot(raw_data$trip_count,raw_data$age,
     xlim = range(1,100),ylim = range(0,165),
     col=ifelse(raw_data$rides_last_week>0,"red","blue"),
     xlab = "Trip Count",ylab="Age",main = "Trip count and Age Scatter",
     sub = "Red = Made Trip last week , Blue = No Trip last week")


plot(raw_data$age,raw_data$rides_last_week,
     xlim = range(1,200),ylim = range(0,50),
     xlab = "Age",ylab="Rides_Last_Week",main = "Rides_Last_Week vs Age in Weeks",
     sub = "Red = Made Trip last week , Blue = No Trip last week")


raw_data[['age']]
age_data = as.vector(raw_data['age'])
hist((raw_data[['age']]))
