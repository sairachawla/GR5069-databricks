# Databricks notebook source
## importing necessary libraries
from pyspark.sql.functions import datediff
from pyspark.sql.functions import current_date
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import avg, upper, max, min, countDistinct, last

# COMMAND ----------

# MAGIC %md #### Read in the datasets

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

df_standings = spark.read.csv('s3://columbia-gr5069-main/raw/driver_standings.csv', header=True)

# COMMAND ----------

display(df_standings)

# COMMAND ----------

# MAGIC %md #### Transform data

# COMMAND ----------

## creating age column for q.4
df_drivers =  df_drivers.withColumn('age', datediff(current_date(), df_drivers.dob)/365)

# COMMAND ----------

df_drivers = df_drivers.withColumn('age', df_drivers.age.cast(IntegerType()))

# COMMAND ----------

## recreating the dataset we joined in class
df_lap_drivers = df_drivers.join(df_laptimes, on=['driverId'])

# COMMAND ----------

df_lap_drivers = df_lap_drivers.join(df_races.select('raceId', 'year', 'name'), on=['raceId'])

# COMMAND ----------

display(df_lap_drivers)

# COMMAND ----------

## joining 2 datasets to answer q.5
df_driver_standings = df_drivers.join(df_standings, on='driverId')

# COMMAND ----------

display(df_driver_standings)

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

## want to know about only those who won each race
winners = df_results[df_results['position'] == 1]

# COMMAND ----------

display(winners)

# COMMAND ----------

## joining dataset to get pitstop times for each winner
df_pitstops_firstplace = winners.join(df_pitstops, on=['raceId', 'driverId'])

# COMMAND ----------

display(df_pitstops_firstplace)

# COMMAND ----------

## finding the avg duration of each driver on each race and sorting by it
avg_stoptime_firstplace = df_pitstops_firstplace.groupby(['raceId', 'driverId']).agg(avg('duration')).orderBy(['avg(duration)'])

# COMMAND ----------

display(avg_stoptime_firstplace)

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

## created a new column age in the transform data section
display(df_lap_drivers)

# COMMAND ----------

max_min_age_race = df_lap_drivers.groupby(['raceId']).agg(max('age'), min('age'))

# COMMAND ----------

display(max_min_age_race)

# COMMAND ----------

# MAGIC %md ###### Q.5
# MAGIC For a given race, which driver has the most wins and losses?

# COMMAND ----------

wins_losses_race = df_driver_standings.groupby(['raceId']).agg(last('driverId'), max('wins'), min('wins'))

# COMMAND ----------

display(wins_losses_race)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ###### Q.6
# MAGIC Continue exploring the data by answering your own question.

# COMMAND ----------

# MAGIC %md My question is: How many drivers are there for each race?

# COMMAND ----------

display(df_lap_drivers)

# COMMAND ----------

num_drivers_race = df_lap_drivers.groupby('raceId').agg(countDistinct('driverId'))

# COMMAND ----------

display(num_drivers_race)

# COMMAND ----------


