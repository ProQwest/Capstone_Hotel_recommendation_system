from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import Rating,ALS


def main(sc):
   
    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    rank = 4
    #load small file
    data_train = sc.textFile("file:///Expedia/data/test_total.csv")
    training_RDD = data_train.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
    
 
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    
     # Save and load model
    model.save(sc, "file:///Expedia/data/modelCollaborativeFilter")             
    
    print 'model created' 
   

if __name__  == "__main__":

    conf = SparkConf().setMaster("local[*]").setAppName("ClusterHotelRecommendationtestALS")
    sc = SparkContext(conf = conf)
    main(sc)
    sc.stop()
