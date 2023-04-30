# Databricks notebook source
# MAGIC %md
# MAGIC # Regression: Predicting Rental Price
# MAGIC
# MAGIC In this notebook, we will use the dataset we cleansed in the previous lab to predict Airbnb rental prices in San Francisco.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use the SparkML API to build a linear regression model
# MAGIC  - Identify the differences between estimators and transformers

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

# COMMAND ----------

filePath = "s3://columbia-gr5069-main/processed/airbnb/listings_clean/"
airbnbDF = spark.read.parquet(filePath)

airbnbDF = airbnbDF.withColumn("bedrooms", airbnbDF["bedrooms"].cast(DoubleType())).withColumn("price", airbnbDF["price"].cast(DoubleType())).withColumn("bathrooms", airbnbDF["bathrooms"].cast(DoubleType()))

# COMMAND ----------

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC When we are building ML models, we don't want to look at our test data (why is that?). 
# MAGIC
# MAGIC Let's keep 80% for the training set and set aside 20% of our data for the test set. We will use the `randomSplit` method [Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).
# MAGIC
# MAGIC **Question**: Why is it necessary to set a seed? What happens if I change my cluster configuration?

# COMMAND ----------

(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)
print(trainDF.cache().count())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's change the # of partitions (to simulate a different cluster configuration), and see if we get the same number of data points in our training set. 

# COMMAND ----------

(trainRepartitionDF, testRepartitionDF) = (airbnbDF
                                           .repartition(24)
                                           .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression
# MAGIC
# MAGIC We are going to build a very simple model predicting `price` just given the number of `bedrooms`.
# MAGIC
# MAGIC **Question**: What are some assumptions of the linear regression model?

# COMMAND ----------

display(trainDF.select("price", "bedrooms","bathrooms"))

# COMMAND ----------

display(trainDF.select("price", "bedrooms","bathrooms").summary())

# COMMAND ----------

display(trainDF.filter(col("price") == 9999))

# COMMAND ----------

# MAGIC %md
# MAGIC There do appear some outliers in our dataset for the price ($9,999 a night??). Just keep this in mind when we are building our models :).
# MAGIC
# MAGIC We will use `LinearRegression` to build our first model [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression).

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = "bedrooms", labelCol = "price")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explain Params
# MAGIC
# MAGIC When you are unsure of the defaults or what a parameter does, you can call `.explainParams()`.

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC What went wrong? Turns out that the Linear Regression **estimator** (`.fit()`) expected a column of Vector type as input.
# MAGIC
# MAGIC We can easily get the values from the `bedrooms` column into a single vector using `VectorAssembler` [Python]((https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler). VectorAssembler is an example of a **transformer**. Transformers take in a DataFrame, and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule based transformations.

# COMMAND ----------

import mlflow

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

with mlflow.start_run(run_name="Saira Linear Regression Model") as run:
  vecAssembler = VectorAssembler(inputCols = ["bedrooms", "bathrooms"], outputCol = "features")
  
  vecTrainDF = vecAssembler.transform(trainDF)
  
  lr = LinearRegression(featuresCol = "features", labelCol = "price")
  
  lrModel = lr.fit(vecTrainDF)
   # Log model
  mlflow.spark.log_model(lrModel, "linear-regression-model")
  
  vecTestDF = vecAssembler.transform(testDF)
  predDF = lrModel.transform(vecTestDF)

  # Instantiate metrics object
  evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price")

  r2 = evaluator.evaluate(predDF, {evaluator.metricName: "r2"})
  print("  r2: {}".format(r2))
  mlflow.log_metric("r2", r2)

  mae = evaluator.evaluate(predDF, {evaluator.metricName: "mae"})
  print("  mae: {}".format(mae))
  mlflow.log_metric("mae", mae)

  rmse = evaluator.evaluate(predDF, {evaluator.metricName: "rmse"})
  print("  rmse: {}".format(rmse))
  mlflow.log_metric("rmse", rmse)

  mse = evaluator.evaluate(predDF, {evaluator.metricName: "mse"})
  print("  mse: {}".format(mse))
  mlflow.log_metric("mse", mse)
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

vecAssembler = VectorAssembler(inputCols = ["bedrooms"], outputCol = "features")

vecTrainDF = vecAssembler.transform(trainDF)

lr = LinearRegression(featuresCol = "features", labelCol = "price")
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

predDF.select("bedrooms", "bathrooms","features", "price", "prediction").show()

# COMMAND ----------

predDF_final = predDF.select('host_is_superhost','cancellation_policy','instant_bookable','host_total_listings_count','neighbourhood_cleansed','zipcode','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','minimum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price','bedrooms_na','bathrooms_na','beds_na','review_scores_rating_na','review_scores_accuracy_na','review_scores_cleanliness_na','review_scores_checkin_na','review_scores_communication_na','review_scores_location_na','review_scores_value_na','prediction')

# COMMAND ----------

display(predDF_final)

# COMMAND ----------

import mysql.connector

# COMMAND ----------

#Execute only once
mydb = mysql.connector.connect(host="sc5119-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com",user="sc5119",password="DnXRKqgkPrP6fGWkNUYg")
mycursor = mydb.cursor()
mycursor.execute("CREATE DATABASE predictions;")

# COMMAND ----------

predDF_final.write.format('jdbc').options(
      url='jdbc:mysql://sc5119-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/predictions',
      driver='com.mysql.jdbc.Driver',
      dbtable='sc5119_gr5069_airbnb_rentals_predictions',
      user='sc5119',
      password='DnXRKqgkPrP6fGWkNUYg').mode('overwrite').save()

# COMMAND ----------

predDF_final_done = spark.read.format("jdbc").option("url", "jdbc:mysql://sc5119-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/predictions") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "sc5119_gr5069_airbnb_rentals_predictions") \
    .option("user", "sc5119").option("password", "DnXRKqgkPrP6fGWkNUYg").load()

# COMMAND ----------

display(predDF_final_done)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE sc5119_predictions;

# COMMAND ----------

predDF_final_done.write.saveAsTable("sc5119_predictions.airbnb_by_neighborhood")

# COMMAND ----------


