run:
	make clean
	sbt package
	cp target/scala-2.11/birdclassifier_2.11-1.0.jar birdclassifier.jar
	spark-submit --class BirdClassifier birdclassfier.jar in/labeled-small.csv output

clean:
	rm -rf output project target *.jar
