import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ronnygeo on 12/1/16.
  */
object BirdClassifier {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Bird Classifier")
     .setMaster("local")
    val sc = new SparkContext(conf)

    //Initializing constants
    var time = System.currentTimeMillis()

    var input: String = "in/unlabeled-small.csv"
    var output:String = "output"


    //if output is specified, use that else default
    if (args.length > 1) {
      input = args(0)
      output = args(1)
    }

    //Loading the input files and getting the adj list
//    val input_df = sc.read.csv(input)


    //        val wiki_data = sc.textFile(input, 50).persist()

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
