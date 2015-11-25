package main.scala

object PetuniaUtils {
  def removeDotToGetWords(wordsTmpArray: Array[String]): Array[String] = {
    if (wordsTmpArray(wordsTmpArray.length - 1).equals(".")) {
			var result = new Array[String](wordsTmpArray.length - 1)
			for (i <- 0 to (wordsTmpArray.length - 1)) {
				result(i) = wordsTmpArray(i);
			}
			return result;
		}
		return wordsTmpArray;
  }
}