# import statements
import os
import time
import gc
import sys
import plotly
import plotly.plotly as py
import plotly.offline
from plotly.graph_objs import *
from pyspark.sql import SparkSession, types, functions
from pyspark.sql.functions import *
import pandas as pd
import elevation_grid as eg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName('weather_plot').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

elevation_schema = types.StructType([
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('date', types.DateType()),
    types.StructField('tmax', types.FloatType())])

def preprocessDF_taskA(inputs):
    dataDF = spark.read.csv(inputs, schema=tmax_schema)
    dataDF.show(200)
    dataDF = dataDF.withColumn("year", substring(col("date"), 1, 4))
    dataDF = dataDF.drop(col("date"))
    dataDF.show(200)
    dataDF_first = dataDF.select(dataDF["station"], dataDF["latitude"], dataDF["longitude"], dataDF["elevation"], dataDF["tmax"], dataDF["year"]).where(dataDF["year"].between(1940, 1960))
    dataDF_first.show(100)
    # check number of rows in DF before grouping
    print(dataDF_first.count())
    dataDF_second = dataDF.select(dataDF["station"], dataDF["latitude"], dataDF["longitude"], dataDF["elevation"], dataDF["tmax"], dataDF["year"]).where(dataDF["year"].between(2005, 2020))
    dataDF_second.show(100)
    # check number of rows in DF before grouping
    print(dataDF_second.count())
    dataDF_first = dataDF_first.select(dataDF["station"], dataDF["latitude"], dataDF["longitude"], dataDF["elevation"], dataDF["tmax"], dataDF["year"]).groupby("latitude", "longitude").agg(avg("tmax").alias('avg_tmax')).dropDuplicates()
    dataDF_first.show(200)
    dataDF_second = dataDF_second.select(dataDF["station"], dataDF["latitude"], dataDF["longitude"], dataDF["elevation"], dataDF["tmax"], dataDF["year"]).groupby("latitude", "longitude").agg(avg("tmax").alias('avg_tmax')).dropDuplicates()
    dataDF_second.show(200)
    # check number of rows in DF after grouping
    print(dataDF_first.count())
    print(dataDF_second.count())
    dataDF_first.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("range1")
    # setting output path
    output_path = 'range1/'
    # creating system command line
    cmd_users = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'first_range.csv'
    # executing system command
    os.system(cmd_users)
    dataDF_second.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("range2")
    # setting output path
    output_path = 'range2/'
    # creating system command line
    cmd_users = 'mv ' + output_path + 'part-*' + '  ' + output_path + 'second_range.csv'
    # executing system command
    os.system(cmd_users)

def plot_taskA():
    # For producing the first map
    DF = spark.read.csv('range1/first_range.csv', header='true')
    DF = DF.withColumn('average_tmax', col('avg_tmax').cast('double'))
    DF = DF.drop("avg_tmax")
    max_value = DF.agg(max(DF["average_tmax"])).head()
    print(float(max_value[0]))
    DF = DF.withColumn('Lat', col('latitude').cast('double'))
    DF = DF.withColumn('Lon', col('longitude').cast('double'))
    DF = DF.drop("latitude", "longitude")
    DF.show(100)
    DF = DF.toPandas()
    # Plotting using plotly- the visualization will be produced in my plotly account
    scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

    data = [ dict(
            type = 'scattergeo',
            lon = DF["Lon"],
            lat = DF["Lat"],
            text = DF["average_tmax"],
            mode = 'markers',
            marker = dict(
                size = 8,
                opacity = 0.8,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale = scl,
                cmin = 0,
                color = DF["average_tmax"],
                cmax = max_value[0],
                colorbar=dict(
                    title="Mean Temperature"
                )
            ))]

    layout = dict(
            title = 'Average Maximum Temperature over the period 1940-1960',
            colorbar = True,
            geo = dict(
                scope='world',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
                ),
            )

    fig = dict( data=data, layout=layout )
    py.plot( fig, validate=False, filename='Average Temperature over the period 1940-1960' )

    #For producing the second map
    DF2 = spark.read.csv('range2/second_range.csv', header='true')
    DF2 = DF2.withColumn('average_tmax', col('avg_tmax').cast('double'))
    DF2 = DF2.drop("avg_tmax")
    max_value1 = DF2.agg(max(DF2["average_tmax"])).head()
    print(float(max_value1[0]))
    DF2 = DF2.withColumn('Lat', col('latitude').cast('double'))
    DF2 = DF2.withColumn('Lon', col('longitude').cast('double'))
    DF2 = DF2.drop("latitude", "longitude")
    DF2.show()
    DF2 = DF2.toPandas()
    # Plotting using plotly- the visualization will be produced in my plotly account
    scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

    data = [ dict(
            type = 'scattergeo',
            lon = DF2["Lon"],
            lat = DF2["Lat"],
            text = DF2["average_tmax"],
            mode = 'markers',
            marker = dict(
                size = 8,
                opacity = 0.8,
                reversescale = True,
                autocolorscale = False,
                symbol = 'circle',
                line = dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale = scl,
                cmin = 0,
                color = DF2["average_tmax"],
                cmax = max_value[0],
                colorbar=dict(
                    title="Mean Temperature"
                )
            ))]

    layout = dict(
            title = 'Average Maximum Temperature over the period 2005-2020',
            colorbar = True,
            geo = dict(
                scope='world',
                showland = True,
                landcolor = "rgb(250, 250, 250)",
                subunitcolor = "rgb(217, 217, 217)",
                countrycolor = "rgb(217, 217, 217)",
                countrywidth = 0.5,
                subunitwidth = 0.5
                ),
            )

    fig = dict( data=data, layout=layout )
    py.plot(fig, validate=False, filename='Average Temperature over the period 2005-2020')

def plot_taskB1(data, model):
    # help(eg)
    # Assign lat and lon values pole to pole, equator to equator
    latitude, longitude = np.arange(-90,90,.5),np.arange(-180,180,.5)
    val = [[float(values[0]),float(values[1]),float(eg.get_elevation(values[0],values[1])),types.datetime.date(2019,1,30),float(0.0)] for values in [[a,b] for a in latitude for b in longitude]]
    DF = spark.createDataFrame(val,schema=elevation_schema)
    # Create a temp table
    predictionDF = model.transform(DF).createOrReplaceTempView("predictionDF")
    predictionDF = spark.sql("SELECT latitude, longitude, prediction from predictionDF")
    # Convert to pandas to produce a heat map using basemap
    FinalDF = predictionDF.toPandas()
    # Configure parameters needed to produce a heat map for maximum temprature spanning across the globe using basemap
    Config = Basemap(projection='robin',lon_0=0,resolution='c')
    Config.drawcoastlines(linewidth=0.5)
    Config.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0],fontsize=7)
    Config.drawmeridians(np.arange(0.,360.,60.),labels=[0,0,0,1],fontsize=7)
    xaxis, yaxis = Config(FinalDF['longitude'].tolist(), FinalDF['latitude'].tolist())
    zaxis = FinalDF['prediction']
    plot = plt.scatter(xaxis,yaxis,c=zaxis.tolist(),cmap='jet')
    plt.title("Global Maximum Temperature Heat Map for 30-1-2019")
    plt.colorbar(mappable=plot, shrink=0.9, label = "Maximum Temperature")
    # Save the plot
    fig = plt.figure(figsize=(10,8))
    plt.show()
    #fig.savefig('GlobalHeatMap.png')

def plot_taskB2(data, model, test):
    prediction = model.transform(test)
    predictionDF = prediction.withColumn("error",(prediction["prediction"]-prediction["tmax"])/100).createOrReplaceTempView("predictionDF")
    FinalDF = spark.sql("SELECT latitude, longitude, error from predictionDF")
    max_value = FinalDF.agg(max(FinalDF["error"])).head()
    FinalDF = FinalDF.toPandas()
    # Configure basemap for plotting scatterplot on globe for regression error estimate
    Config = Basemap()
    Config.drawcoastlines(linewidth=0.5)
    xaxis, yaxis = Config(FinalDF["longitude"], FinalDF["latitude"])
    Config.fillcontinents(color='white',zorder=0)
    plot = Config.scatter(xaxis, yaxis, c=FinalDF["error"],cmap="Blues")
    Config.colorbar(plot, location='right').ax.set_title('Error Percent')
    plt.title("Model- Regression Error Estimate")
    # Save figure and show plot
    fig1 = plt.figure(figsize=(10,8))
    plt.show()
    #fig1.savefig('ModelevaluationPlot.png')

    # Tried- works!!- Plotting using plotly- the visualization will be produced in my plotly account.
    # Alternate method attempted- But is too slow (to many points to render more then 100k) to render and suggestion was to use 'matplotlib instead'!
    # scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    # [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]
    #
    # data = [ dict(
    #         type = 'scattergeo',
    #         lon = FinalDF["latitude"],
    #         lat = FinalDF["latitude"],
    #         text = FinalDF["error"],
    #         mode = 'markers',
    #         marker = dict(
    #             size = 8,
    #             opacity = 0.8,
    #             reversescale = True,
    #             autocolorscale = False,
    #             symbol = 'circle',
    #             line = dict(
    #                 width=1,
    #                 color='rgba(225, 0, 0)'
    #             ),
    #             colorscale = scl,
    #             cmin = 0,
    #             color = FinalDF["error"],
    #             cmax = float(max_value[0]),
    #             colorbar=dict(
    #                 title="Regression Error Scale"
    #             )
    #         ))]
    #
    # layout = dict(
    #         title = 'Model Evaluation- Regression Error Estimate',
    #         colorbar = True,
    #         geo = dict(
    #             scope='world',
    #             showland = True,
    #             landcolor = "rgb(250, 250, 250)",
    #             subunitcolor = "rgb(217, 217, 217)",
    #             countrycolor = "rgb(217, 217, 217)",
    #             countrywidth = 0.5,
    #             subunitwidth = 0.5
    #             ),
    #         )
    #
    # fig = dict( data=data, layout=layout )
    # py.plot( fig, validate=False, filename='Model Regression Error Estimate')

def main(inputs, model_file, test):
    # Get the data
    dataset = spark.read.csv(inputs, schema=tmax_schema)
    # Load the model- model built using weather_train.py
    model = PipelineModel.load(model_file)
    # Create test dataset
    test_data = spark.read.csv(test, schema=tmax_schema)
    # Pre-processing the data to find the average tmax over a period of time
    preprocessDF_taskA(inputs)
    # Task A plot function- uses Plotly.js
    plot_taskA()
    # Task B.1 Plot function- uses matplotlib
    plot_taskB1(dataset, model)
    # Task B.2 plot function- uses Plotly.js
    plot_taskB2(dataset, model, test_data)

if __name__ == '__main__':
    # Get arguements
    inputs = sys.argv[1]
    model_file = sys.argv[2]
    test = sys.argv[3]
    # call main function
    main(inputs, model_file, test)
