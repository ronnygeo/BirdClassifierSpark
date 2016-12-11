import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, RandomForestClassifier}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
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
    var output: String = "output"
    val numPartitions = 25
    val numFolds = 2
    val labelName = "Agelaius_phoeniceus"
    var test: String = null
    val numTrees = 9
    val probDataSample = 0.8

    //if output is specified, use that else default
    if (args.length > 1) {
      input = args(0)
      output = args(1)
      if (args.length > 2) {
        test = args(2)
      }
    }

    //Setting spark configuration
    val conf = new SparkConf()
      .setAppName("Bird Classifier")
//      .setMaster("local[*]")

    //Getting the spark session
    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    //Getting spark Context
    val sc = spark.sparkContext

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._


    //Function to split the string at the particular index and remove the elements in between
    def splitArr(arr: Array[String], s: Int, offset: Int): Array[String] = {
      val (p1, p2) = arr.splitAt(s)
      p1 ++ (p2.drop(offset))
    }

    //Loading the input files and getting the training set
    val inputRDD = sc.textFile(input).map(line => line.split(","))


    //Removing all duplicate columns
    val newRDD = inputRDD.map(arr => splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1), 0, 1))

    // Apply the schema to the RDD
    val inputDF = rdd2DF(spark, newRDD)

    //Writing the intermediate result with unnecessary columns removed
    inputDF.write.format("csv").option("nullValue", "?").option("header", "true").save(output + "/samplingid")

    //Reading the intermediate result from disk and persist
    var autoDF = spark.read.format("csv").option("header", "true").option("nullValue", "?").option("inferSchema", "true").load(output + "/samplingid").repartition(numPartitions)

    //String indexing the LOC_ID field and dropping the column

    //Moving the labels to a new DF
    val labelDF = autoDF.select(labelName).map { v =>
      if (v.get(0).equals("0")) 0.0 else 1.0
    }

    //Process null values
    autoDF = DFnullFix(autoDF, labelName)

    //Initializing the vector assembler to convert the cols to single feature vector
    val assembler = new VectorAssembler().setInputCols(autoDF.columns).setOutputCol("features")
    val featureDF = assembler.transform(autoDF).select("features")

    //Feature scaler to scale the features
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(featureDF)
    val scaledDF = scaler.transform(featureDF).select($"scaled_features".alias("features"))

    //Combine both DFs to one and convert to a new DF
    val zippedRDD = scaledDF.rdd.zip(labelDF.rdd).map { case (features, label) => (features.getAs[Vector](0), label) }
    val data = zippedRDD.toDF("features", "label")

    //Splitting train/test data
    val Array(trainingData, validationData) = data.randomSplit(Array(0.7, 0.3))

    //List to store the models
    var modelsT = new ListBuffer[CrossValidatorModel]()

    //Sample data randomly from the input and pass to numTrees decision tree classifiers
    for (i <- 0 until numTrees) {
      val subTrainDF = trainingData.sample(true, probDataSample)

      //Initializing the Decision Tree Classifier
      val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxBins(300)
      val pipeline = new Pipeline().setStages(Array(dt))

      // Train a LogisticRegression model.
//      val lr = new LogisticRegression()
//        .setMaxIter(10)
//        .setRegParam(0.3)
//        .setElasticNetParam(0.8)
//
//      // Chain indexers and tree in a Pipeline.
//      val pipeline = new Pipeline()
//        .setStages(Array(lr))

      // No parameter search
      val paramGrid = new ParamGridBuilder().build()
      //Initialling cross validation
      val cvEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
      val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(cvEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(numFolds)
      modelsT += cv.fit(trainingData)
    }

    //    val models = sc.parallelize(modelsT.toList)
    val models = modelsT.toList
    val labelRDD = validationData.select("label").rdd.map(row => row.getDouble(0))

    val predictionsRDD = getPredictions(validationData, models)
    val zTestDF = labelRDD.zip(predictionsRDD).toDF("label", "prediction")

    //Take the label and prediction of the test data and get the accuracy.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(zTestDF)
    println("Accuracy for test set = " + accuracy)
    println("Test Error for test set = " + (1.0 - accuracy))


    //////////////////////////////////////////////////////////////////////////////////////////
    //Loading the input files and getting the test set
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
    val TSid = TautoDF.select("SAMPLING_EVENT_ID").map(row => row.getString(0)).repartition(5)
    TautoDF = TautoDF.drop("SAMPLING_EVENT_ID")
    TautoDF = TautoDF.drop(labelName)

    //Fill null values
    TautoDF = DFnullFix(TautoDF,labelName)

    //Initializing the vector assembler to convert the cols to single feature vector
    val Tassembler = new VectorAssembler().setInputCols(TautoDF.columns).setOutputCol("features")
    val TfeatureDF = Tassembler.transform(TautoDF).select("features")
    //Feature scaler to scale the features
    val Tscaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(TfeatureDF)
    val TscaledDF = Tscaler.transform(TfeatureDF).select($"scaled_features".alias("features"))

    val TrfPredictions = getPredictions(TscaledDF, models)
    val TzippedRDD = TrfPredictions.repartition(5).zip(TSid.rdd).map{case (prediction, id) => (id, prediction)}

    val outDF = TzippedRDD.toDF("SAMPLING_EVENT_ID", "SAW_AGELAIUS_PHOENICEUS")
    outDF.coalesce(1).write.format("csv").option("header", "true").save(output+"/testOut")

    //Stopping the spark session
    spark.stop()
  }


  //***************************************************************************
  //HELPER FUNCTIONS
  //Converts an RDD to a DF by creating a schema
  def rdd2DF(spark : SparkSession, ipRDD : RDD[Array[String]]) : DataFrame ={
    val fields = ipRDD.take(1)(0).map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    val header = ipRDD.first()
    val noheadRDD = ipRDD.filter(!_.sameElements(header))

    // Convert records of the RDD (people) to Rows
    val rowRDD = noheadRDD.map(attributes => Row.fromSeq(attributes))

    // Apply the schema to the RDD
    spark.createDataFrame(rowRDD, schema)
  }

  //Remove null values
  def DFnullFix(idf : DataFrame,labelName:String): DataFrame ={
    var df = idf.drop(labelName)

    //Columns with String values
    val x = List("LOC_ID", "COUNTRY","STATE_PROVINCE","COUNTY","COUNT_TYPE","BAILEY_ECOREGION","SUBNATIONAL2_CODE")

    //Fill null values
    df = df.na.fill("__HEREBE_DRAGONS__", x)
    var indexer = new StringIndexer()
    //Indexing all the columns using the StringIndexer
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

  //Gets the predictions from multiple models and gets the majority value
  def getPredictions(validationData: DataFrame, models: List[CrossValidatorModel]): RDD[Double] = {
    //    val modelsT: ListBuffer[CrossValidatorModel] = null
    //    for (i <- 0 until numTrees) {
    //      modelsT += CrossValidatorModel.load("model"+i)
    //    }
    //    val models = modelsT.toList
    val predictions = models.map{model => model.transform(validationData).select("prediction").rdd.map(row => row.getDouble(0))}
    predictions.head.zip(predictions(1)).zip(predictions(2)).map(line => (line._1._1, line._1._2, line._2)).zip(predictions(3)).zip(predictions(4)).map(line => (line._1._1._1, line._1._1._2, line._1._1._3, line._1._2, line._2)).zip(predictions(5)).zip(predictions(6)).map(line => (line._1._1._1, line._1._1._2, line._1._1._3, line._1._1._4, line._1._1._5, line._1._2, line._2)).zip(predictions(7)).zip(predictions(8)).map(line => (line._1._1._1, line._1._1._2, line._1._1._3, line._1._1._4, line._1._1._5, line._1._1._6, line._1._1._7, line._1._2, line._2)).map { line =>
      var count0 = 0
      var count1 = 0
      line.productIterator.foreach(v => if (v == 0.0) count0 += 1 else count1 += 1)
      if (count0 > count1)
        0.0
      else
        1.0
    }
  }


}
