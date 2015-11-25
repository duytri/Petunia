package main.scala

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel
import vn.hus.nlp.sd.SentenceDetector
import vn.hus.nlp.sd.SentenceDetectorFactory
import vn.hus.nlp.tokenizer.TokenizerOptions
import vn.hus.nlp.tokenizer.VietTokenizer
import vn.hus.nlp.utils.FileIterator
import vn.hus.nlp.utils.TextFileFilter;
import scala.util.Properties

class PetuniaMain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ISLab.Petunia")
    val sc = new SparkContext(conf)
    
    //Split and tokenize text data
		var nTokens = 0
		val senDetector = SentenceDetectorFactory.create("vietnamese")
		
		val inputDirPath = ""
		val outputDirPath = ""

		val property = new Properties()
    
    // Load training data in LIBSVM format.
    val data = MLUtils.loadLibSVMFile(sc, "/home/hduser/git/Petunia/Petunia/data/in/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    // Save and load model
    model.save(sc, "/home/hduser/git/Petunia/Petunia/data/out/svmModels.model")
    val sameModel = SVMModel.load(sc, "/home/hduser/git/Petunia/Petunia/data/out/svmModels.model")
  }
}