# Databricks notebook source
## importing necessary libraries
from pyspark.sql.functions import datediff
from pyspark.sql.functions import current_date
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import avg, upper

# COMMAND ----------

# MAGIC %md #### Read in the dataset

# COMMAND ----------

df_laptimes = spark.read.csv('s3://columbia-gr5069-main/raw/lap_times.csv', header=True)

# COMMAND ----------

display(df_laptimes)

# COMMAND ----------

df_drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True)

# COMMAND ----------

display(df_drivers)

# COMMAND ----------

df_races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True)

# COMMAND ----------

display(df_races)

# COMMAND ----------

df_pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header=True)

# COMMAND ----------

display(df_pitstops)

# COMMAND ----------

df_results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True)

# COMMAND ----------

display(df_results)

# COMMAND ----------

# MAGIC %md #### Transform data

# COMMAND ----------

df_drivers =  df_drivers.withColumn('age', datediff(current_date(), df_drivers.dob)/365)

# COMMAND ----------

df_drivers = df_drivers.withColumn('age', df_drivers.age.cast(IntegerType()))

# COMMAND ----------

df_lap_drivers = df_drivers.join(df_laptimes, on=['driverId'])

# COMMAND ----------

df_lap_drivers = df_lap_drivers.join(df_races.select('raceId', 'year', 'name'), on=['raceId'])

# COMMAND ----------

display(df_lap_drivers)

# COMMAND ----------

# MAGIC %md #### Questions

# COMMAND ----------

# MAGIC %md ###### Q.1
# MAGIC What was the average time each driver spent at the pit stop for each race?

# COMMAND ----------

avg_stoptime_driver_race = df_pitstops.groupby(['driverId', 'raceId']).agg(avg('milliseconds'))

# COMMAND ----------

display(avg_stoptime_driver_race)

# COMMAND ----------

# MAGIC %md ###### Q.2
# MAGIC Rank the average time spent at the pit stop in order of who won each race

# COMMAND ----------

temp = df_results[df_results['position'] == 1]

# COMMAND ----------

display(temp)

# COMMAND ----------

df_stoptime_firstplace = avg_stoptime_driver_race.join(temp, on=['raceId', 'driverId'])

# COMMAND ----------

display(df_stoptime_firstplace)

# COMMAND ----------

# MAGIC %md ###### Q.3
# MAGIC Insert the missing code (e.g: ALO for Alonso) for drivers based on the 'drivers' dataset

# COMMAND ----------

df_drivers = df_drivers.withColumn('code', upper(df_drivers.driverRef[0:3]))

# COMMAND ----------

display(df_drivers)

# COMMAND ----------

# MAGIC %md ###### Q.4
# MAGIC Who is the youngest and oldest driver for each race? Create a new column called “Age”

# COMMAND ----------

display(df_lap_drivers)

# COMMAND ----------


