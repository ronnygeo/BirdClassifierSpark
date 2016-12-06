import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.StandardScaler

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
      .getOrCreate()v

    val sc = spark.sparkContext

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "labeled-small.csv"
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

    //Function to split the string at the particular index and remove the elements in between
    def splitArr(arr: Array[String], s:Int, offset:Int) : Array[String] = {
      val (p1, p2) = arr.splitAt(s)
      p1 ++ (p2.drop(offset))
    }

    //Loading the input files and getting the training set
    val inputRDD = sc.textFile(input, 2).map(line => line.split(","))


    val labelRDD = inputRDD.map(arr => arr(26))
    val labelname = labelRDD.take(1)(0)
    val labelDF = labelRDD.filter(!_.equals(labelname)).map(v => !v.equals("0")).toDF(labelname)


    //Removing all duplicate columns
    val newRDD = inputRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 936), 80, 2), 17, 1), 15, 1), 0, 1)
//      splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 80, 2), 17, 1), 15, 1), 0, 1)
    }

    // Generate the schema based on the string of schema
    val fields = newRDD.take(1)(0).map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    val header = newRDD.first()
    val noheadRDD = newRDD.filter(!_.sameElements(header))
    
    // Convert records of the RDD (people) to Rows
    val rowRDD = noheadRDD.map(attributes => Row.fromSeq(attributes))

    // Apply the schema to the RDD
    val inputDF = spark.createDataFrame(rowRDD, schema)
    
    //Writing the intermediate result with unnecessary columns removed
    inputDF.write.format("csv").option("header", "true").save(output+"/samplingid")

    //TODO: Look at loading it directly without writing to csv
    //Reading the intermediate result from disk and persist
    var autoDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/samplingid").cache()

    //String indexing the LOC_ID field and dropping the column
    var indexer = new StringIndexer()

    //One hot encoder
//    var encoder = new OneHotEncoder()

    //Columns with String values
    val x = List("LOC_ID", "COUNTRY","STATE_PROVINCE","COUNTY","COUNT_TYPE","BAILEY_ECOREGION","SUBNATIONAL2_CODE")

    //Fill null values
    autoDF = autoDF.na.fill("__HEREBE_DRAGONS__", x)

    //Indexing all the columns
    for (xname <- x) {
      indexer = new StringIndexer()
      .setInputCol(xname)
      .setOutputCol(s"${xname}_INDEX")
      autoDF = indexer.fit(autoDF).transform(autoDF)
      //Dropping the columns
      autoDF = autoDF.drop(xname)

//      encoder = new OneHotEncoder().setInputCol(s"${xname}_INDEX")
//        .setOutputCol(s"${xname}_VECTOR")
//      autoDF = encoder.transform(autoDF)
    }


    //Removing columns with the null values as it wont help in classification
    val nullCols = autoDF.schema.fields.filter(_.dataType == StringType).map(_.name)
    for(col <- nullCols) {
      autoDF = autoDF.drop(col)
    }

    //Filling null values with 0
    autoDF = autoDF.na.fill(0)

    //Initializing the vector assembler to convert the cols to single feature vector
    val assembler = new VectorAssembler().setInputCols(autoDF.columns).setOutputCol("features")
    val featureDF = assembler.transform(autoDF).select("features")

    //Feature scaler to scale the features
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(featureDF)
    val scaledDF = scaler.transform(featureDF).select($"scaled_features".alias("features"))
    //.map(_.getAs[Vector]("scaled_features").toArray)
    scaledDF.write.parquet(output+"/fDF")


    //    labelsPredictions = data.map(lambda lp: lp.label).zip(predictions)

    val rows = scaledDF.rdd.zip(labelDF.rdd).map{
      case (rowLeft, rowRight) => Row.fromSeq(rowLeft.toSeq ++ rowRight.toSeq)
    }
    //    val newSchema = StructType(scaledDF.schema.fields ++ labelDF.schema.fields)


    //Random split data into training and test


    //Create a list of features


    //Choose m features from the list of features with a probability randomly


    //split the training data into random subsets using probability


    //Pass the subset of features and training data to decision tree classifier


    //Try with multiple classifiers, like logistic regression or SVM or LDA








//    val Y = input_df.select("Agelaius_phoeniceus").map(row => !row(0).equals("0"))
//    val X = input_df.drop(input_df.columns.zipWithIndex.filter(t => (escapCols.contains(t._2))
//      // || duplicateCols.contains(t._2))
//    ).map(t => t._1):_*)

//    X.write.format("csv").save(output+"/samplingid.csv")

//    Y.show(1000)
   /*
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
     * 
     */
    sc.stop()
//    spark.stop()
  }
}
