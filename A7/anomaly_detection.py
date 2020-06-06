# anomaly_detection.py
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as functions
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import operator
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.sql.functions import max
from pyspark.sql.functions import when

# Write your functions here//

# find_distinct_fea function helps to get the values at indices specified beforehand.
def find_distinct_fea(features, pos):

    return features[pos]

# one_hot_encoder function performs the actual one-hot encoding of categorical features.
def one_hot_encoder(features, indices, distinct_fea_list):
    """
        Input: $features represents a DataFrame with two columns: "id" and "rawFeatures"
               $indices represents which dimensions in $rawFeatures are categorical features,
                e.g., indices = [0, 1] denotes that the first two dimensions are categorical features.
               $distinct_fea_list represents the list of unique/distinct values from the dataset(initial dataframe) given.

        Output: Return a encoded one-hot-vector
    """
    len_dis_fea_list = len(distinct_fea_list)
    intial_vec = [0.0]
    len_features = len(features)
    # intialize the one-hot vector with all zeroes.
    encoded_vec = intial_vec*len_dis_fea_list
    print('intialized one-hot vector', encoded_vec)

    # Add 1.0 at the position in the intialized one-hot vector based on the value
    for iter_val in indices:
        pos_val = distinct_fea_list.index(features[iter_val])
        encoded_vec[pos_val] = 1.0
    # get the non-categorical columns in the dataframe
    num_fea_list = []
    for iter_val in range(len_features):
        if iter_val not in indices:
            num_fea_list.append(features[iter_val])

    encoded_vec.extend(num_fea_list)
    print('Encoded One-hot vector', encoded_vec)

    return encoded_vec

# score_func computes the score for datapoint(x) with respect to the min and max clusters as described in the equation.
def score_func(x, n_max_value, n_min_value, prediction):
    """
        Input: $x represents the data point
               $n_max_value size of the largest cluster
               $n_min_value size of the smallest cluster
               $prediction represents the stored list of predictions for all datapoints

        To compute the score of a data point x, we use:

            score(x) = (N_max - N_x)/(N_max - N_min),

        where N_max and N_min represent the size of the largest and smallest clusters, respectively,
        and N_x represents the size of the cluster assigned to x

        Output: Return score value for every datapoint 'x' given as input.
    """
    # To handle divide by zero error, if incase all the clusters have equal number of data points (same size).
    if (n_max_value - n_min_value) == 0:
        print('Divide by Zero error: As all clusters have equal data points (are of equal size)')
        return None
    # comput score for datapoint(x)
    score = float(n_max_value - prediction[str(x)]) / (n_max_value - n_min_value)

    return score


class AnomalyDetection():

    def readData(self, filename):
        self.rawDF = sqlContext.read.parquet(filename).cache()
        temp_DF = self.rawDF
        temp_DF.show()

        # Toy Dataset
        # data = [(0, ["http", "udt", 0.4]), \
        #         (1, ["http", "udf", 0.5]), \
        #         (2, ["http", "tcp", 0.5]), \
        #         (3, ["ftp", "icmp", 0.1]), \
        #         (4, ["http", "tcp", 0.4])]
        # schema = ["id", "rawFeatures"]
        # self.rawDF = spark.createDataFrame(data, schema)
        # temp_DF = self.rawDF
        # temp_DF.show()

    def cat2Num(self, df, indices):
        """
            Input: $df represents a DataFrame with two columns: "id" and "rawFeatures"
                   $indices represents which dimensions in $rawFeatures are categorical features,
                    e.g., indices = [0, 1] denotes that the first two dimensions are categorical features.

            Output: Return a new DataFrame that adds the "features" column into the input $df

            Comments: The difference between "features" and "rawFeatures" is that
            the latter transforms all categorical features in the former into numerical features
            using one-hot key representation
        """
        distinct_fea_list = []
        filtered_list = []
        df.show()
        # Compute the distinct features in the dataframe given
        for iter_val in indices:
            distinct_func = udf(lambda value : find_distinct_fea(value,iter_val), StringType())
            feature_DF = df.select(distinct_func(df.rawFeatures)).distinct()
            print('feature DF')
            feature_DF.show()
            feature_list = feature_DF.collect()
            distinct_fea_list.extend(feature_list)

        print('distinct_fea_list value:', distinct_fea_list)
        # One-hot encode the categorical features in the dataframe(df)
        filtered_list = [row['<lambda>(rawFeatures)'] for row in distinct_fea_list]
        one_hot_encoding = udf(lambda val : one_hot_encoder(val, indices, filtered_list), ArrayType(StringType()))
        final_DF = df.select(df.id, df.rawFeatures, one_hot_encoding (df.rawFeatures)\
                    .alias('features'))
        # Show the final one-hot encoded df
        final_DF.show()

        return final_DF

    def addScore(self, df):
        """
            Input: $df represents a DataFrame with four columns: "id", "rawFeatures", "features", and "prediction"
            Output: Return a new DataFrame that adds the "score" column into the input $df

            To compute the score of a data point x, we use:

                score(x) = (N_max - N_x)/(N_max - N_min),

            where N_max and N_min represent the size of the largest and smallest clusters, respectively,
            and N_x represents the size of the cluster assigned to x
        """
        prediction = {}
        df.show()
        # finding the cluster size, where size means the number of data points present in a cluster
        cluster_count = df.groupBy('prediction').count()
        store_list = cluster_count.collect()
        # store the cluster size obtained as a list
        len_storeslist = len(store_list)
        # find the max and min values of cluster(in terms of its size)
        n_min_value = cluster_count.agg({"count": "min"}).collect()[0][0]
        n_max_value = cluster_count.agg({"count": "max"}).collect()[0][0]
        print("min_value:", n_min_value, "max_value:", n_max_value)

        for iter_val in range(len_storeslist):
            prediction[str(store_list[iter_val]['prediction'])] = store_list[iter_val]['count']
        print(prediction)

        # score the predictions by calling the 'score_func' function.
        score_function = udf(lambda x : score_func(x, n_max_value, n_min_value, prediction), FloatType())
        score = df.withColumn('score',(score_function(df['prediction'])))
        print('score:', score)

        return score

    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show()

        return df3.where(df3.score > t)

if __name__ == "__main__":
    # Create Spark Context
    spark = SparkSession.builder \
            .master("local") \
            .appName("anamoly detecion(A7)") \
            .getOrCreate()
    sc = spark.sparkContext
    # using SQLContext to read parquet file
    sqlContext = SQLContext(sc)
    # Call AnomalyDetection class to perform the one-hot Encoding, K-means clustering and Scroing functions.
    ad = AnomalyDetection()
    ad.readData('data/logs-features-sample')
    print('file read succesfully')
    anomalies = ad.detect(8, 0.97)
    print(anomalies.count())
    anomalies.show()
