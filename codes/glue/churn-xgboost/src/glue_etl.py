import sys
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import split, col, regexp_replace, mean, when, log, lit
from pyspark.ml.feature import StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
import pandas as pd
from pyspark.ml import Pipeline

# Retrieve parameters for the Glue job.
args = getResolvedOptions(
    sys.argv, ["JOB_NAME", "S3_INPUT_FILE", "S3_TRAIN_KEY", "S3_VALIDATE_KEY"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# Create a PySpark dataframe from the source table.
# df = spark.read.csv(args["S3_INPUT_FILE"], format="csv",
#                     inferSchema=True, header=True)
df = spark.read\
    .format("csv")\
    .option('header', 'true')\
    .load(args["S3_INPUT_FILE"])

print('Data loaded from S3: ', df.head())
df.printSchema()

####################  DATA CLEANING  ####################
# Create new CompanyName column
df = df.withColumn("CompanyName", split(col("CarName"), " ")[0])

# Replace incorrect company names
replacement_map = {
    'maxda': 'mazda',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen'
}

for incorrect, correct in replacement_map.items():
    df = df.withColumn('CompanyName', when(col('CompanyName') == incorrect, correct).otherwise(col('CompanyName')))

df = df.withColumnRenamed('maxda', 'mazda') \
    .withColumnRenamed('porcshce', 'porsche') \
    .withColumnRenamed('toyouta', 'toyota') \
    .withColumnRenamed('vokswagen', 'volkswagen') \
    .withColumnRenamed('vw', 'volkswagen')

# for incorrect, correct in replacement_map.items():
#     df = df.withColumn('CompanyName', regexp_replace(
#         'CompanyName', incorrect, correct))

# # Applying log transformation
# scale_col = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize',
#              'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']

# for column in scale_col:
#     df = df.withColumn(column, log(df[column]))

# Drop unnecessary columns
df = df.drop("car_ID", 'CarName', 'symboling','wheelbase','boreratio','stroke','compressionratio','peakrpm')
print('After dropping unnecessary columns: ', df.columns)


# categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
#                     'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'CompanyName']

# indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_cols]
# encoders = [OneHotEncoder(inputCols=[indexer.getOutputCol()], outputCols=[column+"_ohe"]) for indexer, column in zip(indexers, categorical_cols)]

# pipeline = Pipeline(stages=indexers + encoders)
# model = pipeline.fit(df)
# df = model.transform(df)

# # Drop original categorical columns after encoding
# df = df.drop(*categorical_cols)


####################  DATA PREPARATION  ####################
# Group by CompanyName and calculate mean price
# company_price_mean = df.groupBy('CompanyName').agg(
#     mean('price').alias('mean_price')).toDF()
# print(f"Company price mean: {company_price_mean}")

# # Join back with the original dataframe to include mean price
# df = df.join(company_price_mean, on='CompanyName', how='left')
# print(f"Joined company price mean: {df.head()}")

# # Create bins for cars based on mean price
# df = df.withColumn('CarsRange', when(col('mean_price') < 10000, 'Budget')
#                    .when((col('mean_price') >= 10000) & (col('mean_price') < 20000), 'Medium')
#                    .otherwise('Highend'))
# print(f"Added CarsRange column: {df.columns}")

# # Select relevant columns and create dummy variables
# columns_to_keep = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
#                    'enginetype', 'cylindernumber', 'fuelsystem', 'wheelbase', 'carlength',
#                    'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower',
#                    'citympg', 'highwaympg', 'price', 'CarsRange']
# # print(f'Selected columns: {columns_to_keep}')

# catagorical_col = ['fueltype', 'aspiration', 'doornumber', 'carbody',
#                    'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'company']
# df = df.select(*columns_to_keep)
# print(f'Selected columns: {df.columns}')
# df = df.selectExpr(
#     "*", *(f"CAST({c} AS INTEGER)" for c in columns_to_keep if c not in ['price', 'CarsRange']))
# print(f'Selected columns with casting: {df.head()}')
# # Identify duplicate columns
# duplicate_columns = df.columns[df.columns.duplicated(keep=False)]
# print(f"Duplicate columns: {duplicate_columns}")

# Convert categorical values to binary columns
categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'CompanyName']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_columns]

pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

df = df.withColumnRenamed('price', 'label') 

# pandas_df = df.toPandas()
# pandas_df_encoded = pd.get_dummies(columns=["fueltype", "enginelocation", "aspiration", "doornumber", "carbody", "drivewheel", "enginetype",
                                            # "cylindernumber", "fuelsystem"], data=pandas_df)
# print(f"Added dummy variables: {pandas_df_encoded.head()}")

# pandas_df_encoded.columns = [c.replace(' ', '_').replace(
#     '.', '_').replace('-', '_') for c in df.columns]
# print(f"Columns after renaming: {pandas_df_encoded.columns}")

####################  FEATURE SCALING  ####################
# numerical_columns = pandas_df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
# numerical_columns.remove('price')
# # Assemble features
# assembler = VectorAssembler(inputCols=numerical_columns, outputCol="features")
# print(f"Assembled features type: {type(assembler)}")
# # Standardize features
# scaler = StandardScaler(
#     inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# print('scaler: ', type(scaler))

# # Create PySpark DataFrame from Pandas
# sparkDF = spark.createDataFrame(pandas_df_encoded)
# print('spark data: ', sparkDF.head())
# # Transform data
# data = assembler.transform(sparkDF)
# print("Transformed data: ", data.head())

####################  SPLITING DATA  ####################
# Split the dataframe in to training and validation dataframes.
# X = df.drop('price')
# y = df.select('price')

train_data, val_data = df.randomSplit([0.8, 0.2], seed=42)

# Write both dataframes to the destination datastore.
train_path = args["S3_TRAIN_KEY"]
val_path = args["S3_VALIDATE_KEY"]

train_data.write.save(train_path, format="csv", mode="overwrite")
val_data.write.save(val_path, format="csv", mode="overwrite")

# Complete the job.
job.commit()
