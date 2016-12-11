run:
	make clean
	sbt package
	cp target/scala-2.11/birdclassifier_2.11-1.0.jar birdclassifier.jar
	spark-submit --class BirdClassifier birdclassifier.jar in/labeled-train.csv output
	spark-submit --class BirdPredictor birdclassifier.jar cvmodel output in/labeled-test.csv


clean:
	rm -rf output project cvmodel target *.jar
