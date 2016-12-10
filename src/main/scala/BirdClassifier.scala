import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ListBuffer

/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdClassifier {
  def main(args: Array[String]): Unit = {
    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "labeled-train.csv"
    var output:String = "output"
    val numPartitions = 25
    val numFolds = 7
    val labelName = "Agelaius_phoeniceus"
    var test: String = null
    val numTrees = 14


    //TODO: Implement numPartitions while reading data

    //if output is specified, use that else default
    if (args.length > 1) {
      input = args(0)
      output = args(1)
      if (args.length > 2) {
        test = args(2)
      }
    }

    val conf = new SparkConf()
      .setAppName("Bird Classifier")
//     .setMaster("local[4]")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._


    //Function to split the string at the particular index and remove the elements in between
    def splitArr(arr: Array[String], s:Int, offset:Int) : Array[String] = {
      val (p1, p2) = arr.splitAt(s)
      p1 ++ (p2.drop(offset))
    }

    //Loading the input files and getting the training set
    val inputRDD = sc.textFile(input).map(line => line.split(","))


    //Removing all duplicate columns
    val newRDD = inputRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1), 0, 1)
    }

//    val newRDD = inputRDD.mapPartitions{ arr =>
//      var itr: List[Array[String]] = List()
//      while(arr.hasNext) {
//        val data = arr.next()
//        itr = List(splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(data, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1), 0, 1)) ::: itr
//      }
//      itr.iterator
//    }

    // Apply the schema to the RDD
    val inputDF = rdd2DF(spark, newRDD)
    
    //Writing the intermediate result with unnecessary columns removed
    inputDF.write.format("csv").option("nullValue","?").option("header", "true").save(output+"/samplingid")

    //TODO: Look at loading it directly without writing to csv
    //Reading the intermediate result from disk and persist
    var autoDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/samplingid").repartition(numPartitions)

    //TODO: Move pre-processing to a function that takes the required values, as test data also needs to be preprocessed
    //String indexing the LOC_ID field and dropping the column
    //var indexer = new StringIndexer()

    val labelDF = autoDF.select(labelName).map{v =>
      if (v.get(0).equals("0")) 0.0 else 1.0
    }

    autoDF = DFnullFix(autoDF,labelName)

    //Initializing the vector assembler to convert the cols to single feature vector
    val assembler = new VectorAssembler().setInputCols(autoDF.columns).setOutputCol("features")
    val featureDF = assembler.transform(autoDF).select("features")

    //Feature scaler to scale the features
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(featureDF)
    val scaledDF = scaler.transform(featureDF).select($"scaled_features".alias("features"))

    val zippedRDD = scaledDF.rdd.zip(labelDF.rdd).map{case (features, label) => (features.getAs[Vector](0), label)}
    val data = zippedRDD.toDF("features", "label")

    val Array(trainingData, validationData) = data.randomSplit(Array(0.7, 0.3))

    //Using default random forest classifier
    val rfClassifier = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(numTrees)
      .setFeatureSubsetStrategy("0.7")

    val pipeline = new Pipeline().setStages(Array(rfClassifier))

    val paramGrid = new ParamGridBuilder().build() // No parameter search

    val cvEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(cvEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds)

    val rfModel = cv.fit(trainingData)
    val rfPredictions = rfModel.transform(validationData)

    //Take the label and prediction of the test data and get the accuracy.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(rfPredictions)
    println("Accuracy for test set = " + accuracy)
    println("Test Error for test set = " + (1.0 - accuracy))

    //Loading the input files and getting the training set
    val testRDD = sc.textFile(test, numPartitions).map(line => line.split(","))


    //Removing all duplicate columns
    val filteredRDD = testRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1)
    }

    // Apply the schema to the RDD
    val TinputDF = rdd2DF(spark,filteredRDD)
    
    //Writing the intermediate result with unnecessary columns removed
    TinputDF.write.format("csv").option("header", "true").save(output+"/Tsamplingid")

    //TODO: Look at loading it directly without writing to csv
    //Reading the intermediate result from disk and persist
    var TautoDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/Tsamplingid").repartition(numPartitions)

    //TODO: Move pre-processing to a function that takes the required values, as test data also needs to be preprocessed
    //String indexing the LOC_ID field and dropping the column

    val TSid = TautoDF.select("SAMPLING_EVENT_ID")
    TautoDF = TautoDF.drop("SAMPLING_EVENT_ID")
    TautoDF = TautoDF.drop(labelName)

    //Fill null values
    TautoDF = DFnullFix(TautoDF,labelName)

    //Initializing the vector assembler to convert the cols to single feature vector
    val Tassembler = new VectorAssembler().setInputCols(TautoDF.columns).setOutputCol("features")
    val TfeatureDF = Tassembler.transform(TautoDF).select("features")

    val TrfPredictions = rfModel.transform(TfeatureDF)
    
    val TzippedRDD = TrfPredictions.select("prediction").rdd.zip(TSid.rdd).map{case (Row(prediction), Row(id)) => (id.toString(),prediction.toString())}
    TzippedRDD.coalesce(3).saveAsTextFile(output+"/Tout")


    //Stopping the spark session
    spark.stop()
  }

  def rdd2DF(spark : SparkSession,ipRDD : RDD[Array[String]]) : DataFrame ={
    val fields = ipRDD.take(1)(0).map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    val header = ipRDD.first()
    val noheadRDD = ipRDD.filter(!_.sameElements(header))

    // Convert records of the RDD (people) to Rows
    val rowRDD = noheadRDD.map(attributes => Row.fromSeq(attributes))

    // Apply the schema to the RDD
    spark.createDataFrame(rowRDD, schema)
  }

  def DFnullFix(idf : DataFrame,labelName:String): DataFrame ={
    var df = idf.drop(labelName)

    //Columns with String values
    val x = List("LOC_ID", "COUNTRY","STATE_PROVINCE","COUNTY","COUNT_TYPE","BAILEY_ECOREGION","SUBNATIONAL2_CODE")

    //Fill null values
    df = df.na.fill("__HEREBE_DRAGONS__", x)
    var indexer = new StringIndexer()
//    var encoder = new OneHotEncoder()
    //Indexing all the columns
    for (xname <- x) {
      indexer = new StringIndexer()
        .setInputCol(xname)
        .setOutputCol(s"${xname}_INDEX")
      df = indexer.fit(df).transform(df)
    }

    //Removing columns with the null values and Strings as it wont help in classification
    val nullCols = df.schema.fields.filter(_.dataType == StringType).map(_.name)
    for(col <- nullCols) {
      df = df.drop(col)
    }

    //Filling null values with 0
    df.na.fill(0)
  }
  
}
