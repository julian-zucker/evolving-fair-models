package io.evvo

import scala.io.Source

object data {
  /** Represents a dataset.
    *
    * @param name The name of the dataset, used for pretty-printing.
    * @param data The data in the dataset.
    */
  case class DataSet(name: String, data: Seq[LabeledDatapoint]) {
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
      val dataSource = Source.fromFile(f"data/out/${datasetName}.data")
      val labelSource = Source.fromFile(f"data/out/${datasetName}.labels")

      // A shoddy CSV parser that doesn't handle quotes or anything other than
      // straight regex-separated float values.
      val data = dataSource
        .getLines()
        .map(_.split(",") .map(_.toDouble).toVector)
        .toVector

      val labels = labelSource.getLines().map(_.toInt).toVector

      dataSource.close()
      labelSource.close()
      DataSet(datasetName, data.zip(labels).map { case (d, l) => LabeledDatapoint(d, l) })
    }
  }

  /** A labeled datapoint has a set of features and a label. */
  case class LabeledDatapoint(features: DataPoint, label: Int)

  /** A single data point is just a sequence of double-valued features. */
  type DataPoint = IndexedSeq[Double]
}
