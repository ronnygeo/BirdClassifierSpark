import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidatorModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ListBuffer

/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdPredictor {
  def main(args: Array[String]): Unit = {
    //Initializing constants
    var time = System.currentTimeMillis()

    var cvLoc: String = "cvmodel"
    var output:String = "output"
    val numPartitions = 25
    val numFolds = 2
    val labelName = "Agelaius_phoeniceus"
    var test: String = null
    val numTrees = 14

    //if output is specified, use that else default
    if (args.length > 1) {
      cvLoc = args(0)
      output = args(1)
      if (args.length > 2) {
        test = args(2)
      }
    }

    val conf = new SparkConf()
      .setAppName("Bird ClassifierTest")
      .setMaster("local[*]")

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
    val testRDD = sc.textFile(test, numPartitions).map(line => line.split(","))


    //Removing all duplicate columns
    //    val filteredRDD = testRDD.map{arr =>
    //      splitArr(splitArr(splitArr(splitArr(splitArr(arr, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1)
    //    }
    val outheader = splitArr(splitArr(splitArr(splitArr(splitArr(testRDD.take(1)(0), 19, 7), 20, 928), 81, 2), 17, 1), 15, 1)
    val nTRDD = testRDD.filter(!_(0).equals("SAMPLING_EVENT_ID"))

    val filteredRDD = nTRDD.mapPartitions{ arr =>
      var itr: List[Array[String]] = List()
      while(arr.hasNext) {
        val data = arr.next()
        itr = List(splitArr(splitArr(splitArr(splitArr(splitArr(data, 19, 7), 20, 928), 81, 2), 17, 1), 15, 1)) ::: itr
      }
      itr.iterator
    }

    // Apply the schema to the RDD
    val TinputDF = rdd2DF(spark, filteredRDD, outheader)

    //Writing the intermediate result with unnecessary columns removed
    TinputDF.write.format("csv").option("header", "true").save(output+"/Tsamplingid")

    //TODO: Look at loading it directly without writing to csv
    //Reading the intermediate result from disk and persist
    var TautoDF = spark.read.format("csv").option("header", "true").option("nullValue","?").option("inferSchema", "true").load(output+"/Tsamplingid").repartition(numPartitions)

    val TSid = TautoDF.select("SAMPLING_EVENT_ID")
    TautoDF = TautoDF.drop("SAMPLING_EVENT_ID")
    TautoDF = TautoDF.drop(labelName)

    //Fill null values
    TautoDF = DFnullFix(TautoDF,labelName)

    //Initializing the vector assembler to convert the cols to single feature vector
    val Tassembler = new VectorAssembler().setInputCols(TautoDF.columns).setOutputCol("features")
    val TfeatureDF = Tassembler.transform(TautoDF).select("features")

    //Loading the model from previous computation to transform data
    val TrfPredictions = CrossValidatorModel.load("cvmodel").transform(TfeatureDF)

    val TzippedRDD = TrfPredictions.select("prediction").rdd.zip(TSid.rdd).map{case (Row(prediction), Row(id)) => (id.toString(),prediction.toString())}

    val outDF = TzippedRDD.toDF("SAMPLING_EVENT_ID", "SAW_AGELAIUS_PHOENICEUS")
    outDF.coalesce(1).write.format("csv").option("header", "true").save(output+"/testOut")


    //Stopping the spark session
    spark.stop()
  }

  def rdd2DF(spark : SparkSession,ipRDD : RDD[Array[String]], header: Array[String]) : DataFrame ={
    val fields = header.map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    //    val header = ipRDD.first()
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
