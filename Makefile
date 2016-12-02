run:
	make clean
	sbt package
	cp target/birdclassifier-0.1-SNAPSHOT.jar birdclassifier.jar
	spark-submit --class BirdClassifier birdclassfier.jar in/unlabeled-small.csv output

clean:
	rm -rf output project target *.jar
