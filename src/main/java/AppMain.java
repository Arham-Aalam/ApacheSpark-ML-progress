import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.StructType;

public class AppMain {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf()
                .setAppName("Example Spark App")
                .setMaster("local[2]")
                .set("spark.executor.memory","1g");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new SQLContext(sparkContext);


        //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        StructType schema = new StructType()
                .add("PassengerId", "long")
                .add("label", "long")
                .add("Pclass", "long")
                .add("Name", "String")
                .add("Gender", "String")
                .add("Age", "long")
                .add("SibSp", "long")
                .add("Parch", "long")
                .add("Ticket", "String")
                .add("Fare", "float")
                .add("Cabin", "String")
                .add("Embarked", "String");


        Dataset<Row> trainDF = sqlContext
                .read()
                .format("com.databricks.spark.csv")
                .schema(schema)
                .option("Header", "true")
                .load("/home/arham/IdeaProjects/hotwaxdatascience/data/train.csv")
                .na()
                .drop();
        //.select("PassengerId", "label", "Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked")

        //trainDF.show(20);
        //trainDF.printSchema();

        /*

        System.out.println("======== % of male survived\n" + ((float) trainDF.where("Gender='male' and label='1'").count()/ (float) trainDF.count()) * 100);

        System.out.println("======== % of female survived\n" + ((float) trainDF.where("Gender='female' and label='1'").count()/ (float) trainDF.count()) * 100);

        System.out.println("======== % of male not survived\n" + ((float) trainDF.where("Gender='male' and label='0'").count()/ (float) trainDF.count()) * 100);

        System.out.println("======== % of female not survived\n" + ((float) trainDF.where("Gender='female' and label='0'").count()/ (float) trainDF.count()) * 100);

        System.out.println("======== % of people travel in class 1 and survived \n" + (float) trainDF.where("Pclass='1' and label='1'").count()/(float) trainDF.count() * 100);

        System.out.println("======== % of people travel in class 2 and survived \n" + (float) trainDF.where("Pclass='2' and label='1'").count()/(float) trainDF.count() * 100);

        System.out.println("======== % of people travel in class 3 and survived \n" + (float) trainDF.where("Pclass='3' and label='1'").count()/(float) trainDF.count() * 100);

        System.out.println("======== % of people embarked in port C and survived\n " + (float) trainDF.where("Embarked='C' and label='1'").count()/(float) trainDF.count() * 100);

        System.out.println("======== % of people embarked in port S and survived\n " + (float) trainDF.where("Embarked='S' and label='1'").count()/(float) trainDF.count() * 100);

        System.out.println("======== % of people embarked in port Q and survived\n " + (float) trainDF.where("Embarked='Q' and label='1'").count()/(float) trainDF.count() * 100);

        */

        Dataset<Row> train = trainDF.drop("Name", "Ticket");

        StringIndexer genderIndex = new StringIndexer()
                .setInputCol("Gender")
                .setOutputCol("genderIndex")
                .setHandleInvalid("keep");

        /*
        StringIndexer fareIndex = new StringIndexer()
                .setInputCol("Fare")
                .setOutputCol("FareIndex")
                .setHandleInvalid("keep");
        */

        StringIndexer cabinIndex = new StringIndexer()
                .setInputCol("Cabin")
                .setOutputCol("CabinIndex")
                .setHandleInvalid("keep");

        StringIndexer embarkedIndex = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("EmbarkedIndex")
                .setHandleInvalid("keep");

        VectorAssembler assembler = new VectorAssembler().setInputCols(new String[]{ "Pclass", "Age", "SibSp", "Parch", "genderIndex", "Fare", "CabinIndex", "EmbarkedIndex"}).setOutputCol("features");

        LogisticRegression logisticRegression = new LogisticRegression()
                .setMaxIter(30)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] { genderIndex, cabinIndex, embarkedIndex, assembler, logisticRegression });
        PipelineModel model = pipeline.fit(train);


        // Evaluation

        StructType testSchema = new StructType()
                .add("PassengerId", "long")
                .add("Pclass", "long")
                .add("Name", "String")
                .add("Gender", "String")
                .add("Age", "long")
                .add("SibSp", "long")
                .add("Parch", "long")
                .add("Ticket", "String")
                .add("Fare", "float")
                .add("Cabin", "String")
                .add("Embarked", "String");



        Dataset<Row> test = sqlContext
                .read()
                .format("com.databricks.spark.csv")
                .option("Header", "true")
                .schema(testSchema)
                .load("/home/arham/IdeaProjects/hotwaxdatascience/data/test.csv")
                .na()
                .drop();

        Dataset<Row> predictions = model.transform(test.drop("Name", "Ticket"));

        //predictions.show(20);
        predictions.select("rawPrediction").show(20);

        Dataset<Row> passengerPrediction = predictions.withColumn("predictions", predictions.col("prediction").cast("long")).select("PassengerId", "predictions");

        // not survive passengers
        passengerPrediction.where("predictions = '0'").show(20);

        // survived passengers
        passengerPrediction.where("predictions = '1'").show(20);


        /*
        // need label column in "test" data to execute below code

        logistic  evaluator = new BinaryClassificationEvaluator()
                                                                .setRawPredictionCol("rawPrediction");
        evaluator.evaluate(predictions);

        System.out.println("  ============================= \n" + evaluator.getMetricName());

        */
    }
}
