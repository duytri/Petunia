package main.scala

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

object PetuniaUtils {
  def removeDotToGetWords(wordsTmpArray: Array[String]): ArrayBuffer[String] = {
    if (wordsTmpArray(wordsTmpArray.length - 1).equals(".")) {
      var result = new ArrayBuffer[String](wordsTmpArray.length - 1)
      for (i <- 0 to (wordsTmpArray.length - 2)) {
        result(i) = wordsTmpArray(i)
      }
      return result
    }
    var result = new ArrayBuffer[String](wordsTmpArray.length)
    result.insertAll(0, wordsTmpArray)
    return result
  }

  def addOrIgnore(eachWordSet: Map[String, Int], someWords: ArrayBuffer[String]): Unit = {
    val specialChars = Array((" "), ("."), (","), ("\t"), ("..."))
    someWords.foreach { x =>
      {
        if (!specialChars.contains(x)) {
          val y = x.toLowerCase()
          if (!eachWordSet.contains(y))
            eachWordSet += y -> 1
          else eachWordSet.update(y, eachWordSet(y) + 1)
        }
      }
    }
  }
  
  def convert2RDD(rdd: RDD[LabeledPoint], label: LabeledPoint): RDD[LabeledPoint] = {
    rdd.map { x =>
      label
    }
  }
}