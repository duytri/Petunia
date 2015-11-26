package main.scala

import scala.collection.mutable.ArrayBuffer

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
}