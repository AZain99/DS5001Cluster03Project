from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def main():
    spark = SparkSession.builder \
        .appName("Fraud_Detection_Model_Testing") \
        .master("yarn") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    print(">>> Loading pre-trained model asset from HDFS...")
    model_path = "hdfs:///models/fraud_rf_model"
    saved_model = PipelineModel.load(model_path)

    print(">>> Loading independent test dataset from HDFS...")
    test_transaction = spark.read.csv("hdfs:///data/test_transaction.csv", header=True, inferSchema=True)
    test_identity = spark.read.csv("hdfs:///data/test_identity.csv", header=True, inferSchema=True)

    test_data = test_transaction.join(test_identity, on="TransactionID", how="left")

    label_col = 'isFraud'
    if label_col in test_data.columns:
        test_data = test_data.dropna(subset=[label_col])

    print(">>> Executing batch transformations and generating fraud predictions...")
    predictions = saved_model.transform(test_data)

    predictions.select("TransactionID", "TransactionAmt", "rawPrediction", "probability", "prediction").show(10)

    if label_col in predictions.columns:
        print(">>> Calculating model performance statistics...")
        
        evaluator_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator_roc.evaluate(predictions)
        
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
        f1_score = evaluator_f1.evaluate(predictions)

        print("\n==============================================")
        print(f" TEST METRICS SUMMARY")
        print(f" Area Under ROC (AUC): {auc:.4f}")
        print(f" F1-Score:             {f1_score:.4f}")
        print("==============================================\n")
    else:
        print(">>> Predictions completed successfully. No ground truth labels found in test set to evaluate accuracy metrics.")

    spark.stop()

if __name__ == "__main__":
    main()