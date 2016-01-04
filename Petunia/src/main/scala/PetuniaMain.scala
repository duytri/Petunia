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
import vn.hus.nlp.utils.TextFileFilter
import java.util.Properties
import java.io.File
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.collection.mutable.Map
import scala.collection.mutable.MapBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

class PetuniaMain {
  def main(args: Array[String]): Unit = {
    //~~~~~~~~~~Initialization~~~~~~~~~~
    val conf = new SparkConf().setAppName("ISLab.Petunia")
    val sc = new SparkContext(conf)

    //~~~~~~~~~~Split and tokenize text data~~~~~~~~~~
    var nTokens = 0
    val senDetector = SentenceDetectorFactory.create("vietnamese")

    val currentDir = new File(".").getCanonicalPath
    val currentLibsDir = currentDir + File.separator + "libs"

    val inputDirPath = currentDir + File.separator + "data" + File.separator + "in"
    val outputDirPath = currentDir + File.separator + "data" + File.separator + "out"

    val inputDirFile = new File(inputDirPath);

    val property = new Properties()
    property.setProperty("sentDetectionModel", currentLibsDir + File.separator + "models" + File.separator
      + "sentDetection" + File.separator + "VietnameseSD.bin.gz");
    property.setProperty("lexiconDFA", currentLibsDir + File.separator + "models" + File.separator + "tokenization"
      + File.separator + "automata" + File.separator + "dfaLexicon.xml");
    property.setProperty("unigramModel", currentLibsDir + File.separator + "models" + File.separator + "tokenization"
      + File.separator + "bigram" + File.separator + "unigram.xml");
    property.setProperty("bigramModel", currentLibsDir + File.separator + "models" + File.separator + "tokenization"
      + File.separator + "bigram" + File.separator + "bigram.xml");
    property.setProperty("externalLexicon", currentLibsDir + File.separator + "models" + File.separator + "tokenization"
      + File.separator + "automata" + File.separator + "externalLexicon.xml");
    property.setProperty("normalizationRules", currentLibsDir + File.separator + "models" + File.separator
      + "tokenization" + File.separator + "normalization" + File.separator + "rules.txt");
    property.setProperty("lexers", currentLibsDir + File.separator + "models" + File.separator + "tokenization"
      + File.separator + "lexers" + File.separator + "lexers.xml");
    property.setProperty("namedEntityPrefix", currentLibsDir + File.separator + "models" + File.separator
      + "tokenization" + File.separator + "prefix" + File.separator + "namedEntityPrefix.xml");

    val tokenizer = new VietTokenizer(property);
    tokenizer.turnOffSentenceDetection();

    //~~~~~~~~~~Get all input files~~~~~~~~~~
    val inputFiles = FileIterator.listFiles(inputDirFile,
      new TextFileFilter(TokenizerOptions.TEXT_FILE_EXTENSION));
    System.out.println("Tokenizing all files in the directory, please wait...");
    val startTime = System.currentTimeMillis();

    var wordSetByFile = new Array[Map[String, Int]](inputFiles.length) // Map[word, frequency in document]
    //Foreach text file
    for (aFile <- inputFiles) {
      // get the simple name of the file
      val input = aFile.getName()
      // the output file have the same name with the automatic file
      val output = outputDirPath + File.separator + input
      println(aFile.getAbsolutePath() + "\n" + output)
      // tokenize the content of file
      val sentences = senDetector.detectSentences(aFile.getAbsolutePath())
      for (i <- 0 to sentences.length) {
        val words = tokenizer.tokenize(sentences(i))
        val wordsTmpArr = words(0).split(" ")
        PetuniaUtils.addOrIgnore(wordSetByFile(i), PetuniaUtils.removeDotToGetWords(wordsTmpArr))
      }
    }
    //~~~~~~~~~~Calculate TFIDF~~~~~~~~~~
    var tfidfWordSet = new Array[Map[String, (Double, (Int, Double))]](inputFiles.length) // Map[word, (TF-value, (doc no., number of doc contains word))]
    for (i <- 0 to inputFiles.length) {
      for (oneWord <- wordSetByFile(i)) {
        tfidfWordSet(i) += oneWord._1 -> TFIDFCalc.tfIdf(oneWord, i, wordSetByFile)
      }
    }

    //~~~~~~~~~~Remove stopwords~~~~~~~~~~
    //// Load stopwords from file
    val stopwordFilePath = "./libs/vietnamese-stopwords.txt"
    var arrStopwords = new ArrayBuffer[String]
    Source.fromFile(stopwordFilePath, "utf-8").getLines().foreach { x => arrStopwords.append(x) }
    //// Foreach document, remove stopwords
    for (i <- 0 to inputFiles.length) {
      tfidfWordSet(i) --= arrStopwords
    }

    //~~~~~~~~~~Normalize by TFIDF~~~~~~~~~~
    val lowerUpperBound = (0, 1)
    var attrWords = Map[String, (Int, Double)]()
    for (i <- 0 to inputFiles.length) {
      tfidfWordSet(i).foreach(x => {
        val tfidf = x._2._1 * Math.log10(x._2._2._1 / x._2._2._2)
        if (tfidf <= lowerUpperBound._1 || tfidf >= lowerUpperBound._2) {
          tfidfWordSet(i).remove(x._1)
        }
        else attrWords += (x._1 -> (x._2._2._1, x._2._2._2))
      })
    }
    
    //~~~~~~~~~~Create vector~~~~~~~~~~
    var vectorWords = RDD
    vectorWords += LabeledPoint()

    // Sort descending
    //var tfidfResult = HashMap(tfidfResultSet.toSeq.sortWith(_._2 > _._2): _*)

    // Show result
    //tfidfWordSet map (f => println(f._1 + " -------- " + f._2))

    var endTime = System.currentTimeMillis()
    var duration = (endTime - startTime) / 1000
    println(
      "Tokenized " + nTokens + " words of " + inputFiles.length + " files in " + duration + " (s).\n");

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    model.save(sc, outputDirPath + File.separator + "svmModels")
    val sameModel = SVMModel.load(sc, outputDirPath + File.separator + "svmModels")
  }
}