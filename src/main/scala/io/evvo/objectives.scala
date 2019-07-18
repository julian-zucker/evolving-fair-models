package io.evvo

import io.evvo.ctree._
import io.evvo.data.DataSet
import io.evvo.island.population.{Maximize, Minimize, Objective}

/** Holds objectives for ctrees. */
object objectives {

  case class Accuracy()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("Accuracy", Maximize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val correct = dataset.trainData.map(
        dataPoint => dataPoint.label == predict(sol, dataPoint.features))
      correct.count(identity).toDouble / correct.length
    }
  }

  case class FalsePositiveRate()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("FalsePositiveRate", Minimize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val falsePositives = dataset.trainData.map(
        dataPoint => dataPoint.label == 1 && predict(sol, dataPoint.features) == 2)
      falsePositives.count(identity).toDouble / falsePositives.length
    }
  }

  case class FalseNegativeRate()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("FalseNegativeRate", Minimize) {
    override protected def objective(sol: ClassificationTree): Double = {
        val falseNegatives = dataset.trainData.map(
          dataPoint => dataPoint.label == 2 && predict(sol, dataPoint.features) == 1)
        falseNegatives.count(identity).toDouble / falseNegatives.length
      }
  }
}
