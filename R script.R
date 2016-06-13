
#-loading data in R
Data = read.csv2("C:/Expedia/data/part-m-00000.csv", sep=",",
                   comment.char = "",check.names = FALSE,quote="\"" ,stringsAsFactors = FALSE)

str(Data)
Data$orig_destination_distance = as.numeric(Data$orig_destination_distance)

hist(Data$orig_destination_distance, 
     main="Histogram for orig_destination_distance", 
     xlab="orig_destination_distance", 
     border="blue", 
     col="green",
     xlim=c(0,12000),
     las=1, 
     breaks=15 ,prob = TRUE)


hist(Data$hotel_cluster, 
     main="Histogram for hotel_cluster", 
     xlab="hotel_cluster", 
     border="blue", 
     col="green",
     xlim=c(0,100),
     las=1, 
     breaks=100,prob = TRUE)

lines(density(Data$hotel_cluster))


hist(Data$hotel_cluster, 
     main="Histogram for hotel_cluster", 
     xlab="hotel_cluster", 
     border="blue", 
     col="green",
     xlim=c(0,100),
     las=1, 
     breaks=100,prob = TRUE)

hist(Data$site_name, 
     main="Histogram for site_name", 
     xlab="site_name", 
     border="blue", 
     col="green",
     xlim=c(0,60),
     las=1, 
     breaks=10 , prob = TRUE)
lines(density(Data$site_name))

hist(Data$posa_continent, 
     main="Histogram for posa_continent", 
     xlab="posa_continent", 
     border="blue", 
     col="green",
     las=1, 
      prob = TRUE)

hist(Data$user_location_country, 
     main="Histogram for posa_continent", 
     xlab="user_location_country", 
     border="blue", 
     col="green",
     xlim=c(0,250),breaks=10 ,
     prob = TRUE)

hist(Data$is_mobile, 
     main="Histogram for is_mobile", 
     xlab="is_mobile", 
     border="blue", 
     col="green")

hist(Data$is_booking, 
     main="Histogram for is_booking", 
     xlab="is_booking", 
     border="blue", 
     col="green")

#-missing data

Train = Data[complete.cases(Data),]

plot(Train$hotel_cluster,Train$orig_destination_distance , col="blue")

plot(Train$hotel_cluster,Train$user_location_region , col="red")
plot(Train$hotel_cluster,Train$user_id , col="red")

Train_num <- Train[,-c(1,8,12,13),drop=FALSE] 
corrplot(cor(Train_num),method = "ellipse")

