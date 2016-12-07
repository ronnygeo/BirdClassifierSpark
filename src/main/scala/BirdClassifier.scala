import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

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
    import spark.implicits._

    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "labeled-small.csv"
    var output:String = "output"
    val numPartitions = 10
    val labelName = "Agelaius_phoeniceus"


    //TODO: Implement numPartitions while reading data

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
    val inputRDD = sc.textFile(input).map(line => line.split(",")).persist()

  //TODO: Convert all to map partitions
//    var labelRDD = inputRDD.map(arr => arr(26))
//    val labelDF = labelRDD.filter(!_.equals(labelName)).map(v => !v.equals("0")).toDF(labelName).cache()

    //Removing all duplicate columns
    val newRDD = inputRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1), 0, 1)
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
    var autoDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/samplingid").repartition(numPartitions).cache()

    //String indexing the LOC_ID field and dropping the column
    var indexer = new StringIndexer()

    val labelDF = autoDF.select(labelName).map{v =>
      if (v.get(0).equals("0")) 0.0 else 1.0
    }

    autoDF = autoDF.drop(labelName)

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
    val featureDF = assembler.transform(autoDF).select("features").cache()

    //Feature scaler to scale the features
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(featureDF)
    val scaledDF = scaler.transform(featureDF).select($"scaled_features".alias("features"))

    val zippedRDD = scaledDF.rdd.zip(labelDF.rdd).map{case (features, label) => (features.getAs[Vector](0), label)}
    val data = zippedRDD.toDF("features", "label").cache()

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    //Using default random forest classifier
    val rfClassifier = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(10)
    rfClassifier.fit(trainingData)

    val pipeline = new Pipeline().setStages(Array(rfClassifier))

    val paramGrid = new ParamGridBuilder().build() // No parameter search

    val cvEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(cvEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)

    val rfModel = cv.fit(trainingData)
    val rfPredictions = rfModel.transform(testData)

    //Take the label and prediction of the test data and get the accuracy.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(rfPredictions)
    println("Accuracy = " + accuracy)
    println("Test Error = " + (1.0 - accuracy))

    //Stopping the spark session
    spark.stop()
  }
}
