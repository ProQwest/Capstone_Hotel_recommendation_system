from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating ,ALS

import math
import itertools

def main(sc):
   

    #load files
    train_1 = sc.textFile("file:///Expedia/data/train_1.csv")
    training_RDD = train_1.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
    
	#load folds files
    train_2 = sc.textFile("file:///Expedia/data/train_1.csv")
    validation_RDD = train_2.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()


    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1])).cache()
    train_RDD = training_RDD.map(lambda x: (x[0], x[1])).cache()


    #Train model in tain set and cross validation set and choose the best model with 
    # the best RMSE in Cross validation set
    seed = 5L
    iterations = 10
    regularization_parameter = [0.1, 0.5 , 1.0 ] 
    ranks = [4, 8]
    errors = []

    min_error = float('inf')
    best_rank = -1

    for rank, regularization_parameter in itertools.product(ranks, regularization_parameter):
	    #train implicit model in train set  
        model = ALS.trainImplicit(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
        #Predict model in validation set              
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        
        #compute root mean square error in prediction validation set  
        rmse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors.append(rmse)
    
        print 'For rank %s the RMSE is %s' % (rank, rmse)
        if rmse < min_error:
            min_error = rmse
            best_rank = rank


    print "The best model was trained with rank = %d and lambda = %.1f, " % (
    best_rank , regularization_parameter) \
    + "and numIter = %d, and its RMSE on the validation set is %f." % (iterations,
    min_error)

if __name__  == "__main__":

    conf = SparkConf().setAppName("HotelClusterRecommendationsALS")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()
