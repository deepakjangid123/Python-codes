from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
# Importing optimus
import optimus as op
import pandas as pd

# from pprint import pprint

countries = ['Colombia', 'US@A', None, 'Spain', 'Spain']
cities = [None, 'New York', ' Sao polo ', '~Madrid', ' Madrid ']
population = [Vectors.dense([37800000, 1567]), Vectors.dense([19795791, 8967]),
              Vectors.dense([12341418, 86237846632]), Vectors.dense([6489162, 676743]),
              Vectors.dense([6489162, 67674323])]

# Creating Spark Session
spark_session = SparkSession.builder.master("local[*]").appName("Optimus").getOrCreate()
# Data as a dictionary
data = {'countries': countries, 'cities': cities, 'population': population}

# DataFrame
df = pd.DataFrame(data)
df = spark_session.createDataFrame(df)


# Instantiation of DataTransformer class
transformer = op.DataFrameTransformer(df)
print('Original dataFrame:')
transformer.show()


# Clear accents
transformer.clear_accents(columns='*')
print('DataFrame after removing accents from all the columns:')
transformer.show()


# Remove special characters
transformer.remove_special_chars(columns=['cities', 'countries'])
print('DataFrame after removing special characters from all the columns:')
transformer.show()


# Convert strings into Upper-Case
transformer.upper_case(columns='*')
print('DataFrame after applying upper_case function to all the columns:')
transformer.show()


# Replace NA with `Default`
transformer.replace_na("Default", columns=["cities"])
print('DataFrame after replacing `None` values in `cities` column with `Default`:')
transformer.show()


# Apply normalizer on population column
transformer.normalizer("population", p=1.0)
print('DataFrame after applying `normalization` to `population` column:')
transformer.show()


# Removing empty rows based on certain columns
transformer.remove_empty_rows("any")
print("Data after removing rows which have empty entries in any of the columns:")
transformer.show()


# Removing duplicate entries
transformer.remove_duplicates(['countries'])
print("Data after removing duplicate rows based on `countries` column:")
transformer.show()