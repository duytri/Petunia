package main.scala

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
  
  def idf(sentences:Array[Array[String]], term: String):Double={
    var n = 0d
		var docsSize = 0
		for (sentence <- sentences) {
			docsSize+=1
			for (word <- sentence if (term.equalsIgnoreCase(word))) {
				n+=1
				//break
			}
		}
		return Math.log10(docsSize / n);
  }
}