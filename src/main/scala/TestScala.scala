import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.StandardScaler

object TestScala {

  def createSparkSession() : SparkSession = {
    return SparkSession.builder.appName("Yelp Help").getOrCreate()
  }

  def createSparkContext() : SparkContext = {
    val conf = new SparkConf().setAppName("YelpHelp Context")
    conf.setMaster("local[2]")
    val sc = new SparkContext(conf)
    sc
  }

  def readData(spark: SparkSession) : DataFrame = {
    val reader = spark.read
    reader.option("delimeter", "\t").
      option("header", "true").
      option("inferSchema","true")
    return reader.csv("/Users/raghavnyati/Desktop/dataset/restaurants_data.csv")//.
      //toDF("business_id", "review_count", "stars", "checkin_count", "city", "state", "categories")
  }

  def creatingFeatureVector(df_indexed: DataFrame) : DataFrame = {
    val assembler = new VectorAssembler().
      setInputCols(Array("review_count", "checkin_count","city", "state")).
      setOutputCol("features")
    assembler.transform(df_indexed)
  }

  def createRandomForestModel() : RandomForestClassifier = {
    val rf = new RandomForestClassifier().setLabelCol("stars").
      setFeaturesCol("scaledFeatures").
      setImpurity("gini").setMaxDepth(20).setNumTrees(500).setFeatureSubsetStrategy("auto")
    rf
  }

  def printDataframe(df: DataFrame) = {
    df.collect().foreach(println)
  }

  def main(args: Array[String]): Unit = {
//    val conf = new SparkConf()
//    conf.setAppName("Yelp Test")
//    conf.setMaster("local[2]")
//    val sc = new SparkContext(conf)
//    println(sc)

    Logger.getLogger("org").setLevel(Level.ERROR)
    val log = LogManager.getRootLogger
    log.info("Start")

    val sc = createSparkContext()
    val spark = createSparkSession()
    log.info("Spark Context and session created successfully")
    import spark.implicits._
    val data = readData(spark)
    log.info("File read success.")

    //printDataframe(data)
    data.printSchema()
    val df_assembled = creatingFeatureVector(data).select("features", "stars")
    df_assembled.show(10)

    val scaler = new StandardScaler().setInputCol("features").
      setOutputCol("scaledFeatures").
      setWithStd(true).
      setWithMean(true)

    val transformedDF =  scaler.fit(df_assembled).transform(df_assembled)

    val Array(trainingData, testData) = transformedDF.randomSplit(Array(0.7, 0.3))
    log.info("Training and testing data created successfully.")

    // Train model. This also runs the indexers.
    val model = createRandomForestModel().fit(trainingData)
    log.info("Model trained success.")

    println(model.featureImportances)

    // Make predictions.
    val predictions = model.transform(testData)
    log.info("Applying model on test data success.")

    val temp_df1 = predictions.select("stars", "prediction")
    val temp_rdd1 = temp_df1.as[(Double,Double)].rdd
    val temp_metrics = new MulticlassMetrics(temp_rdd1)
    val labels = temp_metrics.labels

    val confusionMatrix = sc.parallelize(temp_metrics.confusionMatrix.toArray)

    println(confusionMatrix)

    log.info("End")
    spark.stop()
    sc.stop()
  }
}
