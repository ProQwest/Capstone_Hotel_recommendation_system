import sys
from os.path import isfile
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS,Rating


def main(sc):

    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    rank = 4


    data = sc.textFile("file:///Expedia/data/train1.csv")
    
    ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
    
    new_data = sc.textFile("file:///Expedia/data/new_set.csv")
    
    new_ratings = new_data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
    new_ratings_for_predict_RDD = new_ratings.map(lambda x: (x[0], x[1])).cache()
    
    complete_data = ratings.union(new_ratings).cache()
    
    new_ratings_model = ALS.trainImplicit(complete_data, rank, seed=seed, 
                              iterations=iterations, lambda_=regularization_parameter)
                              
    
    # that not work need more invistigation                        
    #predictions = new_ratings_model.predictAll(0,'83').collect()
    predictions = new_ratings_model.predictAll(new_ratings_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2])).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:2]
    
    recommendations.take(5)
 
if __name__  == "__main__":
    conf = SparkConf().setAppName("Movie_Recommendations")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()