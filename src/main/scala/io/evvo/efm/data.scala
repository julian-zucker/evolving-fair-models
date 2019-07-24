package io.evvo.efm

import io.evvo.efm.ctree.Label

import scala.io.Source

object data {
  case class DataSet(
      name: String,
      trainData: Seq[LabeledDatapoint],
      testData: Seq[LabeledDatapoint]) {
    require(trainData.forall(datapoint =>
      testData.forall(_.features.length == datapoint.features.length)))

    def featureValues(feature: Int): Seq[Double] = this.trainData.map(_.features(feature))

    def randomLabel(): Label = possibleLabels(util.Random.nextInt(possibleLabels.length))

    val numFeatures: Int = trainData.head.features.length
    val possibleLabels: Seq[Int] = trainData.map(_.label).distinct

    override def toString: String = f"DataSet[$name]"
  }

  object DataSet {

    /** Reads a Dataset.
      *
      * @param datasetName The name of the dataset. Determines which files will be read:
      *                    `datasetName.labels` will contain the labels, and `datasetName.data`
      *                    will contain a CSV of the features.
      * @return the CSV at the specified filename.
      */
    def load(datasetName: String): DataSet = {
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

        val labels = labelSource.getLines().map(_.toInt).toVector
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
  case class LabeledDatapoint(features: DataPoint, label: Int, privileged: Boolean)

  /** A single data point is just a sequence of double-valued features. */
  type DataPoint = IndexedSeq[Double]
}
