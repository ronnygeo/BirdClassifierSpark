import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassifier}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

import scala.util.Random
import scala.collection.mutable.ListBuffer


/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Bird Classifier")
     .setMaster("local[*]")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    //Initializing constants
    var time = System.currentTimeMillis()

    var trainInput: String = "labeled-train.csv"
    var testInput: String = "labeled-test.csv"
    var output:String = "output"

    val labelName = "Agelaius_phoeniceus"
    val numPartitions = 10
    val numTrees = 10
  val numDataSample = 0.3

    //TODO: Implement numPartitions while reading data

    //if output is specified, use that else default
    if (args.length > 1) {
      trainInput = args(0)
      testInput = args(1)
      if (args.length > 2) output = args(2)
    }

    //Function to split the string at the particular index and remove the elements in between
    def splitArr(arr: Array[String], s:Int, offset:Int) : Array[String] = {
      val (p1, p2) = arr.splitAt(s)
      p1 ++ (p2.drop(offset))
    }

    //Loading the input files and getting the training set
    val inputRDD = sc.textFile(trainInput).map(line => line.split(",")).persist()

    //Removing all duplicate columns
    val newRDD = inputRDD.map{arr =>
      splitArr(splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1), 0, 1)
//      splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 936), 80, 2), 17, 1), 15, 1), 0, 1)
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
    var trainDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/samplingid").cache()

    //String indexing the LOC_ID field and dropping the column
    var indexer = new StringIndexer()

    //Columns with String values
    val x = List("LOC_ID", "COUNTRY","STATE_PROVINCE","COUNTY","COUNT_TYPE","BAILEY_ECOREGION","SUBNATIONAL2_CODE")


    //Fill null values
    trainDF = trainDF.na.fill("__HEREBE_DRAGONS__", x)
    val numFeatures = trainDF.columns.length - 1
    var featureCols = trainDF.columns.filter(!_.equals(labelName))

    //Indexing all the columns
    for (xname <- x) {
      indexer = new StringIndexer()
      .setInputCol(xname)
      .setOutputCol(s"${xname}_INDEX")
      trainDF = indexer.fit(trainDF).transform(trainDF)
      //Dropping the columns
//      trainDF = trainDF.drop(xname)
    }

    //Choose m features from the list of features with a probability randomly
    var cols = new ListBuffer[List[String]]()
    while(featureCols.length > numFeatures/numTrees || featureCols.length > 0) {
      var sampleCols: List[String] = Random.shuffle(featureCols.toList).take((numFeatures / numTrees) + 1)
      cols += (sampleCols ++ List(labelName))
      featureCols = featureCols.filter(v => !sampleCols.contains(v) || v.equals(labelName))
    }
    val colset = cols.toList

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

    var modelsT = new ListBuffer[CrossValidatorModel]()

    for (cols <- colset) {
      //split the training data into random subsets using probability
      var subDF = trainDF.select(cols.head, cols.tail: _*).sample(true, numDataSample)
      //Creating the labelDF
      val labelDF = subDF.select(labelName)
      subDF.drop(labelName)

      //Removing columns with the null values as it wont help in classification
      val nullCols = subDF.schema.fields.filter(_.dataType == StringType).map(_.name)

      for(col <- nullCols) {
        subDF = subDF.drop(col)
      }

      //Filling null values with 0
      subDF = subDF.na.fill(0)

      //Initializing the vector assembler to convert the cols to single feature vector
      val assembler = new VectorAssembler().setInputCols(subDF.columns).setOutputCol("features")
      val featureDF = assembler.transform(subDF).select("features").cache()

      //Feature scaler to scale the features
      val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaled_features").fit(featureDF)
      val scaledDF = scaler.transform(featureDF).select($"scaled_features".alias("features"))

      val zippedRDD = scaledDF.rdd.zip(labelDF.rdd).map{case (features, label) => (features.getAs[Vector](0), if (label.getString(0).equals("0")) 0.0 else 1.0)}

      //Converting the joined RDD to DF
      val data = zippedRDD.toDF("features", "label").cache()

      // Chain indexers and tree in a Pipeline.
      val pipeline = new Pipeline().setStages(Array(dt))

      // Train model. This also runs the indexers.
      val dtModel = pipeline.fit(data)

      val paramGrid = new ParamGridBuilder().build() // No parameter search

      val cvEvaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

      val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(cvEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5)

      modelsT += cv.fit(data)
    }

    val models = modelsT.toList

    //Testing phase
    //Reading the intermediate result from disk and persist
    var testDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load("testTemp")

    //Fill null values
    testDF = testDF.na.fill("__HEREBE_DRAGONS__", x)
    val numFeatures = testDF.columns.length - 1
    var featureCols = testDF.columns.filter(!_.equals(labelName))

    //Indexing all the columns
    for (xname <- x) {
      indexer = new StringIndexer()
        .setInputCol(xname)
        .setOutputCol(s"${xname}_INDEX")
      testDF = indexer.fit(testDF).transform(testDF).cache()
    }

      //line with Row object and
      var predictionsTotal = ListBuffer[Array[Int]]()
    //Creating the labelDF


    for (i <- 0 until numTrees) {
      val cols = colset(i)
      var subDF = testDF.select(cols.head, cols.tail: _*)
      val labelDF = subDF.select(labelName)
      subDF.drop(labelName)

      //Removing columns with the null values as it wont help in classification
      val nullCols = subDF.schema.fields.filter(_.dataType == StringType).map(_.name)

      for(col <- nullCols) {
        subDF = subDF.drop(col)
      }

      //Filling null values with 0
      subDF = subDF.na.fill(0)

      //Initializing the vector assembler to convert the cols to single feature vector
      val assembler = new VectorAssembler().setInputCols(subDF.columns).setOutputCol("features")
      val featureDF = assembler.transform(subDF).select("features").cache()

      predictionsTotal += models(i).transform(featureDF).select("prediction").collect().map(_(0) == 1).map(v => if (v) 1 else 0)
    }

    for (i <- 0 until testDF.count.toInt) {
      var preds: Array[Boolean] = null
      for (j <- predictionsTotal.indices) {
        preds += predictionsTotal(j)(i)
      }
      preds.map()
    }


    //Pass the test data to the all the models and
//
////    sc.parallelize(, numPartitions)
//
//
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println("Accuracy = " + accuracy)
//    println("Test Error = " + (1.0 - accuracy))
//






    //parallelize these multiple trees

    //Try with multiple classifiers, like logistic regression or SVM or LDA


    //Combine results from multiple classifiers
    //If classification take majority count, else take mean

    //Stopping the spark session
    spark.stop()
  }
}
