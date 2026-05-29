from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

def main():
    spark = SparkSession.builder \
        .appName("Fraud_Detection_Model_Training") \
        .master("yarn") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()

    print(">>> Loading training data from HDFS...")
    train_transaction = spark.read.csv("hdfs:///data/train_transaction.csv", header=True, inferSchema=True)
    train_identity = spark.read.csv("hdfs:///data/train_identity.csv", header=True, inferSchema=True)

    train_data = train_transaction.join(train_identity, on="TransactionID", how="left")

    print(">>> Subsetting features for training...")
    feature_cols = ['TransactionAmt', 'card1', 'card2', 'card3', 'C1', 'C2', 'C3']
    label_col = 'isFraud'

    train_data = train_data.dropna(subset=[label_col])

    print(">>> Building Feature Engineering Pipeline...")
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{c}_imputed" for c in feature_cols]).setStrategy("median")
    imputed_cols = [f"{c}_imputed" for c in feature_cols]

    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="features")

    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=50, maxDepth=10, seed=42)

    pipeline = Pipeline(stages=[imputer, assembler, rf])

    print(">>> Training the distributed Random Forest Model on YARN...")
    model = pipeline.fit(train_data)

    print(">>> Saving the trained model asset back to HDFS...")
    model_path = "hdfs:///models/fraud_rf_model"
    model.write().overwrite().save(model_path)
    print(f">>> Model successfully saved to {model_path}")
    print("Training Done")

    spark.stop()

if __name__ == "__main__":
    main()
