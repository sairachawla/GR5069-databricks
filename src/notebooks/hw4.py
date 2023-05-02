# Databricks notebook source
## importing necessary libraries and functions
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col
from pyspark.ml.regression import RandomForestRegressor
import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import pandas as pd

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Read in datasets
# MAGIC ###### Exploring potential relationships for modeling

# COMMAND ----------

df = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Clean/Preprocess data

# COMMAND ----------

df = df.na.drop("any")

# COMMAND ----------

df = df.withColumn('points', df['points'].cast(DoubleType())).withColumn('position', df['position'].cast(DoubleType())).withColumn('milliseconds', df['milliseconds'].cast(DoubleType()))

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train/Test Split

# COMMAND ----------

(trainDF, testDF) = df.select(['position', 'points', 'milliseconds']).randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Regression
# MAGIC Predicting points from position.

# COMMAND ----------

display(trainDF)

# COMMAND ----------

display(trainDF.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Q1

# COMMAND ----------

# build model using pipline
assembler = VectorAssembler(inputCols=['position','milliseconds'], outputCol='features')
rf = RandomForestRegressor(featuresCol = "features", labelCol = "points", numTrees=5)
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(trainDF)

# COMMAND ----------

# model metrics
predictions = model.transform(testDF)
evaluator = RegressionEvaluator(labelCol='points', metricName='rmse')
rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment Runs

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Q2&3

# COMMAND ----------

with mlflow.start_run(run_name="Saira RF numTrees=500,maxDepth=5") as run:
    # Set the model parameters
    num_trees = 500
    max_depth = 5

    # Log the model parameters
    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_depth", max_depth)

    vecAssembler = VectorAssembler(inputCols=["position"], outputCol="features")

    vecTrainDF = vecAssembler.transform(trainDF)

    rf = RandomForestRegressor(featuresCol="features", labelCol="points", numTrees=num_trees, maxDepth=max_depth)

    rfModel = rf.fit(vecTrainDF)

    # Log model
    mlflow.spark.log_model(rfModel, "random-forest-model")

    vecTestDF = vecAssembler.transform(testDF)
    predDF = rfModel.transform(vecTestDF)

    # Instantiate metrics object
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="points")

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

    # Generate and log artifacts
    importance_df = rfModel.featureImportances
    importance_df = pd.DataFrame(importance_df.toArray(), columns=["importance"])
    importance_df.to_csv("importance.csv", index=False)
    mlflow.log_artifact("importance.csv")

    predictions_df = predDF.select("position", "points", "prediction").toPandas()
    predictions_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.csv")
    
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id

    print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Q4

# COMMAND ----------

# MAGIC %md
# MAGIC My best model was a random forest model with numTrees equal to 500 and maxDepth equal to 5. I made this decision by evaluating based on the metrics: mae, mse, and r2. This model had the lowest mae and mse, and the highest r2. I found that shifting max depth to be greater than 5 did not lead to significant improvements while lowering max depth hindered performance metrics. I found that as I increased numTrees, it performed like an exponential curve on improvement of model metrics. This means that when numTrees is lower, shifts that increase numTrees improve performance metrics more than when numTrees is already at a high amount. 

# COMMAND ----------


