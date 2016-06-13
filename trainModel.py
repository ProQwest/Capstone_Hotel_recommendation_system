from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating ,ALS

import math
import itertools


conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext(conf = conf)
sc.setCheckpointDir('checkpoint')


#load small file
data = sc.textFile("file:///Expedia/data/test_total.csv")
data_header = data.take(1)[0]
ratings = data.filter(lambda line: line!=data_header).map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()


#split training sample in validation and test data sets
training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1])).cache()
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1])).cache()

numTraining = training_RDD.count()
numValidation = validation_for_predict_RDD.count()
numTest = test_RDD.count()
print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)


#Train model in tain set and cross validation set and choose the best model with 
# the best RMSE in Cross validation set
seed = 5L
iterations = 10
regularization_parameter = [0.1, 0.5 , 1.0 ] 
ranks = [4, 8]
errors = []
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
best_regul_parameter = -1
for rank, regularization_parameter in itertools.product(ranks, regularization_parameter):
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
                      
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    
    rmse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors.append(rmse)
    
    print 'For rank %s the RMSE is %s and lambda = %.1f, ' % (rank, rmse, regularization_parameter )
    if rmse < min_error:
        min_error = rmse
        best_rank = rank
        best_Model = model
        bestLambda = regularization_parameter
        bestNumIter = iterations
        best_regul_parameter = regularization_parameter
print "The best model was trained with rank = %d and lambda = %.1f, " % (
best_rank , regularization_parameter) \
+ "and numIter = %d, and its RMSE on the validation set is %f." % (iterations,
min_error)


predictions = best_Model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)
sc.stop()