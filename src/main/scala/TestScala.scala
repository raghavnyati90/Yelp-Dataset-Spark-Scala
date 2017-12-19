import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

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
    val rf = new RandomForestClassifier().setLabelCol("label").
      setFeaturesCol("features").
      setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto")
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
  }
}
