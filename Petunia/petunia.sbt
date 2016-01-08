name := "Petunia project"
version := "1.0"
organization := "uit.islab"
scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.1",
  "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided"
)

unmanagedBase <<= baseDirectory { base => base / "libs" }

mainClass in Compile := Some("main.scala.PetuniaMain")