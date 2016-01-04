package main.scala

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map

object TFIDFCalc {
  def tf(term: String, doc: Map[String, Int]): Double = {
    var wordCount = 0d
    doc.foreach((x => wordCount += x._2))
    doc(term) / wordCount
  }

  def idf(term: String, allDocs: Array[Map[String, Int]]): (Int, Double) = {
    var n = 0d
    allDocs.foreach(x => {
      if (x.contains(term)) n += 1
    })

    return (allDocs.length -> n)
  }

  def tfIdf(word: (String, Int), docIndex: Int, allDocs: Array[Map[String, Int]]): (Double, (Int, Double)) = {
    val term = word._1
    val doc = allDocs(docIndex)
    return (tf(term, doc) -> idf(term, allDocs))
  }
}