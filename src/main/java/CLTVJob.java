
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionSummary;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import scala.Function2;
import scala.collection.Seq;
import scala.collection.Seq$;
import scala.runtime.BoxedUnit;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CLTVJob {

	public static SparkContext sc;
	public static SQLContext sqlContext;
	public static SparkSession spark;
	public static LinearRegressionModel lrModel;

	public static void main(String[] args) {
		
		LogManager.getLogger("org").setLevel(Level.OFF);
		
		spark = SparkSession.builder().appName("JavaLinearRegressionWithElasticNetExample")
				.master("local[*]").getOrCreate();

		sc = spark.sparkContext();
		sqlContext = spark.sqlContext();


/*
		generateTrainingData();
*/

		// Loading Generated Data
		Dataset<Row>  data = spark.read().format("csv").option("header", "true").option("inferSchema", "true").load("data/customerHistory.csv");
		data.show(20);
		data.printSchema();


		// Splitting data.
		Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 2);

		Dataset<Row> training = splits[0];
		Dataset<Row> test = splits[1];

		//Assembling Training Data
		VectorAssembler assembler = new VectorAssembler();
		assembler.setInputCols(new String[] { "MONTH_1", "MONTH_2", "MONTH_3", "MONTH_4"
				, "MONTH_5", "MONTH_6"}).setOutputCol("features");

		Dataset<Row> vectorDataTops = assembler.transform(training).drop("CUST_ID");
		vectorDataTops.show();

		// preparing LR Model
		LinearRegression lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).
				setFeaturesCol("features").setLabelCol("CLV") ;
		lr.setPredictionCol("predictions");

		// fitting the model
		lrModel = lr.fit(vectorDataTops);

		// Print the coefficients and intercept for linear regression.
		System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

		System.out.println("======================================== Training Results =====================================");
		// Summarize the model over the training set and print out some metrics.
		LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
		System.out.println("numIterations: " + trainingSummary.totalIterations());
		System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
		trainingSummary.residuals().show();
		System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
		System.out.println("r2: " + trainingSummary.r2());


	/*
		// assembling test Data
		VectorAssembler vectorAssemblerTest = new VectorAssembler();
		vectorAssemblerTest.setInputCols(new String[]{"MONTH_1", "MONTH_2", "MONTH_3", "MONTH_4", "MONTH_5", "MONTH_6"}).setOutputCol("features");
		Dataset<Row> testData = vectorAssemblerTest.transform(test);


		Dataset<Row> predictions = lrModel.transform(testData);
		predictions.show(20);

		// evaluation through regression evaluator
		RegressionEvaluator regressionEvaluator = new RegressionEvaluator().setMetricName("rmse");
		//double rmse = regressionEvaluator.evaluate(predictions);

		LinearRegressionSummary evaluateResult = lrModel.evaluate(testData);

		// test results
		System.out.println("======================================== Test Results =====================================");
		evaluateResult.residuals().show();
		System.out.println("RMSE: " + evaluateResult.rootMeanSquaredError());
		System.out.println("r2: " + evaluateResult.r2());
		*/


		// Testing with less features having zero values
		Dataset<Row> testingLessFeatures = data.limit(150);

		Dataset<Row> testDataLessFeatures = testingLessFeatures
				.withColumn("MONTH_4_new", testingLessFeatures.col("MONTH_4").multiply(0))
				.withColumn("MONTH_5_new", testingLessFeatures.col("MONTH_5").multiply(0))
				.withColumn("MONTH_6_new", testingLessFeatures.col("MONTH_6").multiply(0))
				.drop(testingLessFeatures.col("CLV"))
				.withColumn("CLV", testingLessFeatures.col("MONTH_1").plus(testingLessFeatures.col("MONTH_2")).plus(testingLessFeatures.col("MONTH_3")).multiply(15));


		testDataLessFeatures.show();

		VectorAssembler vectorAssemblerTest = new VectorAssembler();
		vectorAssemblerTest.setInputCols(new String[]{"MONTH_1", "MONTH_2", "MONTH_3", "MONTH_4_new", "MONTH_5_new", "MONTH_6_new"}).setOutputCol("features");
		Dataset<Row> testData = vectorAssemblerTest.transform(testDataLessFeatures);


		System.out.println("======================================== Test Results with less features having zero =====================================");

		Dataset<Row> predictions = lrModel.transform(testData);
		//predictions.show(20);
		predictions.show(false);


		// Test Summary
		LinearRegressionSummary lrSummary = lrModel.evaluate(testData);

		// test results
		//System.out.println("r2: " + regressionEvaluator.evaluate(predictions));
		System.out.println("RMSE : " + lrSummary.rootMeanSquaredError());



		//****************************  Testing with new generated data ***********************************

		Dataset<Row> unseenData = generateDF(1, 200);

		VectorAssembler vectorAssemblerTestForUnseenData = new VectorAssembler();
		vectorAssemblerTestForUnseenData.setInputCols(new String[]{"MONTH_1", "MONTH_2", "MONTH_3", "MONTH_4", "MONTH_5", "MONTH_6"}).setOutputCol("features");
		Dataset<Row> unseenVectorTestData = vectorAssemblerTestForUnseenData.transform(unseenData);

		Dataset<Row> unseenDataPredictions = lrModel.transform(unseenVectorTestData);
		unseenDataPredictions.show(false);

		LinearRegressionSummary unseenTestDataSummary = lrModel.evaluate(unseenVectorTestData);

		// unseen test results
		System.out.println("RMSE : " + unseenTestDataSummary.rootMeanSquaredError());

		makePrediction(new double[]{20.0, 80.0, 40.0, 120.0, 180.0, 40.0});

		makePrediction(new double[]{10.0, 10.0, 10.0, 10.0, 10.0, 10.0});

		makePrediction(new double[]{45.0, 10.0, 80.0, 0.0, 0.0, 0.0});

		makePrediction(new double[]{0.0, 0.0, 0.0, 120.0, 0.0, 40.0});

		spark.stop();
	}

	static Dataset<Row> generateDF(int countStart, int countEnd) {

		StructType structType = new StructType(
				new StructField[]{
				new StructField(
						"CUST_ID",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_1",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_2",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_3",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_4",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_5",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"MONTH_6",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				),
				new StructField(
						"CLV",
						DataTypes.IntegerType,
						false,
						Metadata.empty()
				)
				}
		);

		Random random = new Random();

		List<Row> rows = new ArrayList<Row>();
		//rows.add(RowFactory.create(1, "v1"));
		//rows.add(RowFactory.create(2, "v2"));

		for(int row=countStart;row<=countEnd; row++) {
			int m1 = random.nextInt(201);
			int m2 = random.nextInt(201);
			int m3 = random.nextInt(201);
			int m4 = random.nextInt(201);
			int m5 = random.nextInt(201);
			int m6 = random.nextInt(201);
			int clv = (m1 + m2 + m3 + m4 + m5 + m6) * 15;
			rows.add(RowFactory.create(
					row,
					m1,
					m2,
					m3,
					m4,
					m5,
					m6,
					clv
			));
		}

		return spark.createDataFrame(rows, structType);
	}

	public static void dataGe() {
		StructField[] structFields = new StructField[]{
				new StructField("intColumn", DataTypes.IntegerType, true, Metadata.empty()),
				new StructField("stringColumn", DataTypes.StringType, true, Metadata.empty())
		};

		StructType structType = new StructType(structFields);

		List<Row> rows = new ArrayList<Row>();
		rows.add(RowFactory.create(1, "v1"));
		rows.add(RowFactory.create(2, "v2"));

		Dataset<Row> df = spark.createDataFrame(rows, structType);

		df.show(20);
	}

	public static void generateTrainingData() {
		// Generating Data
		//dataGe();
		Dataset<Row>  data = spark.read().format("csv").option("header", "true").option("inferSchema", "true").load("data/history.csv");
		Dataset<Row> rawData = generateDF(1101, 1600);
		//rawData.show(20);
		Dataset<Row> finalRaw = data.union(rawData);
		//finalRaw.repartition(1).write().csv("data/customerHistory");
		finalRaw
				.repartition(1)
				.write()
				.format("com.databricks.spark.csv")
				.option("header", "true")
				.save("data/customerHistory");

		finalRaw.show(20);
	}

	public static void makePrediction(double[] featureValues) {

		double predictValue = lrModel.predict(new DenseVector(featureValues));
		System.out.println("Your Prediction : " + predictValue);
	}
}
