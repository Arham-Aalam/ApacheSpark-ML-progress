import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.*;

public class AppDecisionTree {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf()
                .setAppName("Example Spark App")
                .setMaster("local[2]")
                .set("spark.executor.memory","1g");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new SQLContext(sparkContext);

        Dataset<Row> data = sqlContext
                .read()
                .format("csv")
                .option("Header", "true")
                .option("InferSchema", "true")
                .load("data/train.csv")
                .na()
                .drop();

        data.count();
        //data.show(20);
        //data.printSchema();

        Dataset<Row> trainingData = data.select("PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked")
                .withColumn("label",  data.col("Survived"));

        StringIndexer genderIndexedModel = new StringIndexer()
                .setInputCol("Sex")
                .setOutputCol("IndexedSex");

        StringIndexer cabinIndexedModel = new StringIndexer()
                .setInputCol("Cabin")
                .setOutputCol("IndexedCabin");

        StringIndexer EmbarkedIndexedModel = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("IndexedEmbarked");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"PassengerId", "Pclass", "IndexedSex", "Age", "SibSp", "Parch", "Fare", "IndexedCabin", "IndexedEmbarked"})
                .setOutputCol("features");

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier()
                .setFeaturesCol("features")
                .setPredictionCol("SurvivedPrediction");

        Pipeline dTreePipeline = new Pipeline()
                .setStages(new PipelineStage[]{genderIndexedModel, cabinIndexedModel, EmbarkedIndexedModel, assembler, decisionTreeClassifier});

        // Train model. This also runs the indexers.
        PipelineModel model = dTreePipeline.fit(trainingData);

        Dataset<Row> tData = sqlContext
                .read()
                .format("csv")
                .option("Header", "true")
                .option("InferSchema", "true")
                .load("data/test.csv")
                .na()
                .drop();

        tData.count();
        // Make predictions.
        Dataset<Row> predictions = model.transform(tData);
    }
}