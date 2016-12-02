import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Bird Classifier")
     .setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "in/unlabeled-small.csv"
    var output:String = "output"


    //if output is specified, use that else default
    if (args.length > 1) {
      input = args(0)
      output = args(1)
    }

    //Loading the input files and getting the training set
    val input_df = sqlContext.read
      .format("csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(input)

    //TODO: Drop duplicate columns
    input_df.drop("SAMPLING_EVENT_ID").select("CAUS_FIRST_AUTUMN_32F_MEAN").write.format("parquet").save(output+"/samplingid.parquet")

    time = System.currentTimeMillis()

    //Getting the initial N value
    // var N = adj.count
    println("Preprocessing Time: " + (System.currentTimeMillis() - time)+"ms")

    // wiki_data.unpersist()
    //Starting the iterator value
    // var i = 0

    //Stopping the spark context
    sc.stop()
  }
}
