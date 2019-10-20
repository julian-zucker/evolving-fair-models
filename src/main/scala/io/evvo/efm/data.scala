package io.evvo.efm

import io.evvo.efm.ctree.{ClassificationTree, Label, Negative, Positive}

import scala.io.Source

object data {
  case class DataSet(
      name: String,
      trainData: Seq[LabeledDatapoint],
      testData: Seq[LabeledDatapoint]) {
    require(trainData.forall(datapoint =>
      testData.forall(_.features.length == datapoint.features.length)))

    def featureValues(feature: Int): Seq[Double] = this.trainData.map(_.features(feature))

    def randomLabel(): Label = Vector(Positive, Negative)(util.Random.nextInt(2))

    val numFeatures: Int = trainData.head.features.length

    override def toString: String = f"DataSet[$name]"
  }

  object DataSet {
    def loadGerman(): DataSet = this.load("German")
    def loadCompas(): DataSet = this.load("COMPAS")
    def loadAdultIncome(): DataSet = this.load("adult_income")


    /** Reads a Dataset.
      *
      * @param datasetName The name of the dataset. Determines which files will be read:
      *                    `datasetName.labels` will contain the labels, and `datasetName.data`
      *                    will contain a CSV of the features.
      * @return the CSV at the specified filename.
      */
    private def load(datasetName: String): DataSet = {
      def readDataMatrix(dir: String): Seq[LabeledDatapoint] = {
        val dataSource = Source.fromFile(f"data/${dir}/${datasetName}.data")
        val labelSource = Source.fromFile(f"data/${dir}/${datasetName}.labels")
        val privSource = Source.fromFile(f"data/${dir}/${datasetName}.priv")

        // A shoddy CSV parser that doesn't handle quotes or anything other than
        // straight regex-separated float values.
        val data = dataSource
          .getLines()
          .map(_.split(",").map(_.toDouble).toVector)
          .toVector

        // "1" is the positive label, so should be  tree
        val labels = labelSource
        .getLines()
          .map(l =>
            if (l == "Positive") {
              Positive
            } else {
              assert(l == "Negative")
              Negative
            })
          .toVector

        // "True" is privileged
        val priv = privSource.getLines().map(_ == "True").toVector
        require(data.length == labels.length && data.length == priv.length)

        dataSource.close()
        labelSource.close()
        privSource.close()
        data.zip(labels).zip(priv).map { case ((d, l), p) => LabeledDatapoint(d, l, p) }
      }

      DataSet(datasetName, readDataMatrix("train"), readDataMatrix("test"))
    }
  }

  /** A labeled datapoint has a set of features and a label. */
  case class LabeledDatapoint(features: DataPoint, label: Label, isPrivileged: Boolean) {
    def predictedCorrectlyBy(c: ClassificationTree): Boolean = {
      ctree.predict(c, this.features) == this.label
    }

    def predictionFrom(c: ClassificationTree): Label = {
      ctree.predict(c, this.features)
    }
  }

  /** A single data point is just a sequence of double-valued features. */
  type DataPoint = IndexedSeq[Double]
}
