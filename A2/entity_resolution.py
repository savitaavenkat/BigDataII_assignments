# entity_resolution.py
import re
import operator
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import col, udf, concat, lit, split, lower
from pyspark.sql.functions import *
from pyspark.sql.types import StringType

def remove_stopwords(row, stopwords):
    """
    Input: $row represents row of 'joinKey' in a DataFrames
           $stopwords represents the stopwords read from stopwords.txt

    Output: Return tokens after converting them to lowercase and any stopwords encountered

    """
    # Split the input row of tokens
    tokens = re.split(r'\W+', row)
    # Remove stopwords from the tokens and any unicode characters
    tokens = [t.lower() for t in tokens if t not in stopwords and t!= u'']

    return tokens

def spread_joinKey_tokens(row):
    """
    Input: $row represents row of 'joinKey' in a DataFrames

    Output: Return row (id, token)

    """
    # Get the entire list of tokens for a particular row as a string so as to spread each token as a separate row.
    tokens_string = re.search('\[(.*?)\]', row[1]) # I use search here against match to look over the entire string and not just the start.
    # Split the string on a comma so that we get each token separately
    split_tokens = tokens_string.group(1).split(', ')

    return [(row[0], token) for token in split_tokens]

def compute_jaccard_similarity(key):
    """
    Input: $key represents row of that has values = 'joinKey1' + 'joinKey2' of a DataFrame

    Output: Return jaccard_similarity-> which is the jaccard similarity (0,1] value of the sets

    """
    # Get the entire list of tokens (joinKey1 + joinKey2) for a particular row as as a comma separated string
    tokens_string = re.search('\[(.*?)\],\[(.*?)\]', str(key))
    # Check for empty tokens (empty string/ None type)
    if tokens_string is None:
        return 0
    # Split the string on a comma so that we get tokens of joinKey1 and joinKey2 separately and convert them into a 'set'
    tokens_df1 = set(tokens_string.group(1).split(', '))
    tokens_df2 = set(tokens_string.group(2).split(', '))
    if len(tokens_df1) == 0 or len(tokens_df2) == 0:
        return 0
    # Compute the jaccard similarity as (intersection of two sets)/(union of two sets)
    intersection_jaccard = len(tokens_df1 & tokens_df2)#len([token for token in tokens_df1 if token in tokens_df2])
    print('common tokens or intersection of the sets', intersection_jaccard)
    union_jaccard = len(tokens_df1) + len(tokens_df2) - intersection_jaccard
    if union_jaccard == 0:
        return 0
    jaccard_similarity = intersection_jaccard / union_jaccard

    return jaccard_similarity

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = spark.read.parquet(dataFile1).cache()
        self.df2 = spark.read.parquet(dataFile2).cache()
        self.df1.show()
        self.df2.show()

    def preprocessDF(self, df, cols):
        """
        Input: $df represents a DataFrame
               $cols represents the list of columns (in $df) that will be concatenated and be tokenized

        Output: Return a new DataFrame that adds the "joinKey" column into the input $df

        Comments: The "joinKey" column is a list of tokens, which is generated as follows:
                 (1) concatenate the $cols in $df;
                 (2) apply the tokenizer to the concatenated string
        Here is how the tokenizer should work:
                 (1) Use "re.split(r'\W+', string)" to split a string into a set of tokens
                 (2) Convert each token to its lower-case
                 (3) Remove stop words
        """
        stopWordsBC = self.stopWordsBC
        # Remove stopwords (obtained from stopwords.txt) from the tokens
        remove_stopwords_udf = udf(lambda row: remove_stopwords(row, stopWordsBC))
        # Concatenate the specified columns of the input dataframe and then send it udf function to transform joinKey
        preprocessDF_df = df.withColumn("joinKey", remove_stopwords_udf(concat_ws(' ', df[cols[0]], df[cols[1]])))
        preprocessDF_df.show()

        return preprocessDF_df

    def filtering(self, df1, df2):
        """
        Input: $df1 and $df2 are two input DataFrames, where each of them
               has a 'joinKey' column added by the preprocessDF function

        Output: Return a new DataFrame $candDF with four columns: 'id1', 'joinKey1', 'id2', 'joinKey2',
                where 'id1' and 'joinKey1' are from $df1, and 'id2' and 'joinKey2'are from $df2.
                Intuitively, $candDF is the joined result between $df1 and $df2 on the condition that
                their joinKeys share at least one token.

        Comments: Since the goal of the "filtering" function is to avoid n^2 pair comparisons,
                  you are NOT allowed to compute a cartesian join between $df1 and $df2 in the function.
                  Please come up with a more efficient algorithm (see hints in Lecture 2).
                  NOTE: I have filtered the empty tokens in the verification step before computing jaccard similarity (comments given)
        """
        # Distribute the tokens from 'joinKey' column in the dfs to avoid n^2 comparisons and take advantage of spark's shufflehashjoin
        filtering_rdd1 = df1.select(df1.id, df1.joinKey).rdd.map(spread_joinKey_tokens)
        filtering_rdd1 = filtering_rdd1.flatMap(lambda token: token)
        filtering_rdd2 = df2.select(df2.id, df2.joinKey).rdd.map(spread_joinKey_tokens)
        filtering_rdd2 = filtering_rdd2.flatMap(lambda token: token)
        # Convert the RDDs returned as a result of the previous operation to DataFrame
        filtering_df1 = spark.createDataFrame(filtering_rdd1, ('id1', 'token1'))
        filtering_df2 = spark.createDataFrame(filtering_rdd2, ('id2', 'token2'))
        filtering_df1.show()
        filtering_df2.show()
        # Manipulate DataFrame to get the final candDF with columns 'id1', 'joinKey1', 'id2', 'joinKey2'
        mergedDF = filtering_df2.join(filtering_df1, filtering_df1.token1 == filtering_df2.token2, 'inner').select(filtering_df1.id1, filtering_df2.id2).dropDuplicates()
        merged_df1 = mergedDF.join(df1, df1.id == mergedDF.id1, 'inner').select(mergedDF.id1, df1.joinKey.alias('joinKey1'), mergedDF.id2)
        merged_df2 = merged_df1.join(df2, df2.id == merged_df1.id2, 'inner').select(merged_df1.id1, merged_df1.joinKey1, merged_df1.id2, df2.joinKey.alias('joinKey2'))
        merged_df2.show()

        return merged_df2

    def verification(self, candDF, threshold):
        """
            Input: $candDF is the output DataFrame from the 'filtering' function.
                   $threshold is a float value between (0, 1]

            Output: Return a new DataFrame $resultDF that represents the ER result.
                    It has five columns: id1, joinKey1, id2, joinKey2, jaccard

            Comments: There are two differences between $candDF and $resultDF
                      (1) $resultDF adds a new column, called jaccard, which stores the jaccard similarity
                          between $joinKey1 and $joinKey2
                      (2) $resultDF removes the rows whose jaccard similarity is smaller than $threshold
        """
        # Combine joinKey1 and joinKey2 coulmn values of candDF dataframe
        resultDF_combine = candDF.withColumn('jaccard', concat_ws(',', candDF.joinKey1, candDF.joinKey2))
        resultDF_combine.show()
        # Fuction to compute jaccard similarity
        jaccard_udf = udf(lambda row: compute_jaccard_similarity(row))
        # Pass the combined new column to the udf function to compute the jaccard similarity
        resultDF = resultDF_combine.select(resultDF_combine.id1, resultDF_combine.joinKey1, resultDF_combine.id2, resultDF_combine.joinKey2, jaccard_udf(resultDF_combine['jaccard']).alias('jaccard'))
        resultDF.show()

        return resultDF.where(resultDF.jaccard >= threshold)


    def evaluate(self, result, groundTruth):
        """
            Input: $result is a list of matching pairs identified by the ER algorithm
                   $groundTrueth is a list of matching pairs labeld by humans

            Output: Compute precision, recall, and fmeasure of $result based on $groundTruth, and
                    return the evaluation result as a triple: (precision, recall, fmeasure)

        """
        # Compute |R| value-> number of similar records as computed by the algorithm
        R = float(len(result))
        # Compute |T| value-> number of similar tokens computed by algorithm that are orginally present in the groundtruth (human computed) file
        T = float(len([token for token in result if token in groundTruth]))
        # Calculate precision as |T|/|R|
        precision = T / R
        # Compute |A| value-> the number of similar tokens in groundtruth file
        A = float(len(groundTruth))
        # Compute Recall as |T|/|A|
        recall = T / A
        # Calculate the denominator of Fmeasure
        self.number1 = precision + recall
        # Calculate the numerator for Fmeasure
        self.number2 = 2 * precision * recall
        # Compute Fmeasure, if denominator is zero fmeasure = 0 and print divide by zero exception
        if self.number1 != 0:
            fmeasure = self.number2 / self.number1
        else:
            print('DivideByZeroException: You are trying to divide by zero, fmeasure cannot be computed')

        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print ("savitaav Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()))

        candDF = self.filtering(newDF1, newDF2)
        print ("savitaav After Filtering: %d pairs left" %(candDF.count()))

        resultDF = self.verification(candDF, threshold)
        print ("savitaav After Verification: %d similar pairs" %(resultDF.count()))

        return resultDF

if __name__ == "__main__":
    conf = SparkConf().setAppName('EntityResolution')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.appName('ER').getOrCreate()
    er = EntityResolution("Amazon", "Google", "stopwords.txt")
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.rdd.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = spark.read.parquet("Amazon_Google_perfectMapping_sample").rdd.map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print ("savitaav (precision, recall, fmeasure) = ", er.evaluate(result, groundTruth))
