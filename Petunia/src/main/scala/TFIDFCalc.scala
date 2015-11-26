package main.scala

import scala.collection.mutable.ArrayBuffer

object TFIDFCalc {
  def tf(sentence: Array[String], term: String): Double = {
    var termAppear = 0
    var senSize = 0
    for (word <- sentence) {
      senSize += 1
      if (term.equalsIgnoreCase(word))
        termAppear += 1
    }
    return termAppear / sentence.length
  }

  def idf(sentences: Array[ArrayBuffer[String]], term: String): Double = {
    var n = 0d
    var docsSize = 0
    for (sentence <- sentences) {
      docsSize += 1
      var tmpSen = sentence.map { x => x.toLowerCase() }
      sentence.contains(term)
    }
    return Math.log10(docsSize / n);
  }

  def tfIdf(doc: Array[String], docs: Array[ArrayBuffer[String]], term: String): Double = {
    return tf(doc, term) * idf(docs, term)
  }
}