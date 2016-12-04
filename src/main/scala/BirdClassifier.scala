import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Bird Classifier")
     .setMaster("local")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext

    // For implicit conversions like converting RDDs to DataFrames
//    import spark.implicits._

    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "in/labeled-small2.csv"
    var output:String = "output"
    val escapCols = 19 to 952
    val duplicateCols = Seq(1016, 1017)

    //1016, 1017 duplicate
    //19:954
    //Drop: 0, 15, 17


    // String fields: 1, 9:12, 959

    //Class label: 26


    //if output is specified, use that else default
    if (args.length > 1) {
      input = args(0)
      output = args(1)
    }

    //Loading the input files and getting the training set
    val inputRDD = sc.textFile(input, 2).map(line => line.split(","))

    //    val input_df = spark.read
    //      .format("csv")
    //      .option("header", "true") // Use first line of all files as header
    //      .option("inferSchema", "true") // Automatically infer data types
    //      .load(input)



    def splitArr(arr: Array[String], s:Int, offset:Int) : Array[String] = {
      val (p1, p2) = arr.splitAt(s)
      p1 ++ (p2.drop(offset))
    }

    val labelRDD = inputRDD.map(arr => arr(26))

    val newRDD = inputRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 936), 80, 2), 17, 1), 15, 1), 0, 1)
    }

    // Generate the schema based on the string of schema
    val fields = newRDD.take(1)(0).map(
      fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    // Convert records of the RDD (people) to Rows
    val rowRDD = newRDD.map(attributes => Row.fromSeq(attributes))

    // Apply the schema to the RDD
    val inputDF = spark.createDataFrame(rowRDD, schema)

//    // Creates a temporary view using the DataFrame
//    inputDF.createOrReplaceTempView("data")
//
//    // SQL can be run over a temporary view created using DataFrames
//    val results = spark.sql("SELECT * FROM data")

    inputDF.write.format("csv").save(output+"/samplingid.csv")


//    val Y = input_df.select("Agelaius_phoeniceus").map(row => !row(0).equals("0"))
//    val X = input_df.drop(input_df.columns.zipWithIndex.filter(t => (escapCols.contains(t._2))
//      // || duplicateCols.contains(t._2))
//    ).map(t => t._1):_*)

//    X.write.format("csv").save(output+"/samplingid.csv")

//    Y.show(1000)
    time = System.currentTimeMillis()

    //Getting the initial N value
    println("Preprocessing Time: " + (System.currentTimeMillis() - time)+"ms")

    // Create a LogisticRegression instance. This instance is an Estimator.
    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    // We may set parameters using setter methods.
    lr.setMaxIter(10)
      .setRegParam(0.01)

    // Learn a LogisticRegression model. This uses the parameters stored in lr.
//    val model1 = lr.fit(X)



    //Stopping the spark session
    sc.stop()
//    spark.stop()
  }
}
