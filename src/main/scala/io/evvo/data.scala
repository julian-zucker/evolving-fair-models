package io.evvo

import io.evvo.ctree.Label

import scala.io.Source

object data {
  case class DataSet(name: String, trainData: Seq[LabeledDatapoint], testData: Seq[LabeledDatapoint]) {
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

        // A shoddy CSV parser that doesn't handle quotes or anything other than
        // straight regex-separated float values.
        val data = dataSource
          .getLines()
          .map(_.split(",").map(_.toDouble).toVector)
          .toVector

        val labels = labelSource.getLines().map(_.toInt).toVector

        dataSource.close()
        labelSource.close()
        data.zip(labels).map { case (d, l) => LabeledDatapoint(d, l) }
      }

      DataSet(datasetName, readDataMatrix("train"), readDataMatrix("test"))
    }
  }

  /** A labeled datapoint has a set of features and a label. */
  case class LabeledDatapoint(features: DataPoint, label: Int)

  /** A single data point is just a sequence of double-valued features. */
  type DataPoint = IndexedSeq[Double]
}
