name := "Petunia project"
version := "1.0"
organization := "uit.islab"
scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.1",
  "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided",
  "vn.hus.nlp.tokenizer" % "tokenizer" % "4.1.1" from "file://home/hduser/git/Petunia/Petunia/libs/vn.hus.nlp.tokenizer-4.1.1.jar",
  "vn.hus.nlp.sd" % "sd" % "2.0.0" from "file://home/hduser/git/Petunia/Petunia/libs/vn.hus.nlp.sd-2.0.0.jar",
  "vn.hus.nlp.utils" % "utils" % "1.0.0" from "file://home/hduser/git/Petunia/Petunia/libs/vn.hus.nlp.utils-1.0.0.jar"
)