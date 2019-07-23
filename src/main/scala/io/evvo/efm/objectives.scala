package io.evvo.efm

import io.evvo.efm.ctree._
import io.evvo.efm.data.DataSet
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


  case class FalseNegativeRateRatio()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("FalseNegativeRateRatio", Minimize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val falseNegativesP = dataset.trainData
        .filter(_.privileged)
        .map(dataPoint => dataPoint.label == 2 && predict(sol, dataPoint.features) == 1)
      val privFpRate = falseNegativesP.count(identity).toDouble / falseNegativesP.length

      val falseNegativesNotP = dataset.trainData
        .filter(!_.privileged)
        .map(dataPoint => dataPoint.label == 2 && predict(sol, dataPoint.features) == 1)
      val notPrivFpRate = falseNegativesNotP.count(identity).toDouble / falseNegativesNotP.length

      // Regardless of who is winning, if one is higher than the other, it's a problem.
      math.max(1, math.max(privFpRate / notPrivFpRate, notPrivFpRate / privFpRate))
    }
  }

  /** Measures the ratio at which the privileged and non-priveleged groups are given label 1. */
  case class DisparateImpact()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("DisparateImpact", Minimize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val privPreds = dataset.trainData
        .filter(_.privileged)
        .map(_.features)
        .map(predict(sol, _))

      val nonprivPreds = dataset.trainData
        .filter(!_.privileged)
        .map(_.features)
        .map(predict(sol, _))

      val ppRatio = privPreds.count(_ == 1).toDouble / privPreds.length
      val nppRatio = nonprivPreds.count(_ == 1).toDouble / nonprivPreds.length
      math.max(1, math.max(ppRatio / nppRatio, nppRatio / ppRatio))
    }
  }
}
