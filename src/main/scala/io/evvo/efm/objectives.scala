package io.evvo.efm

import io.evvo.efm.ctree._
import io.evvo.efm.data.{DataSet, LabeledDatapoint}
import io.evvo.island.population.{Maximize, Minimize, Objective}

/** Holds objectives for ctrees. */
object objectives {

  case class Accuracy()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("Accuracy", Maximize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val correctCount =
        dataset.trainData.count(dataPoint => dataPoint.label == predict(sol, dataPoint.features))
      correctCount.toDouble / dataset.trainData.length
    }
  }

  case class FalsePositiveRate()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalsePositiveRate", Minimize) {
    private val negativeLabelData = dataset.trainData.filter(_.label == 2)

    override protected def objective(sol: ClassificationTree): Double = {
      val falsePositives =
        negativeLabelData.count(dataPoint => predict(sol, dataPoint.features) == 1)
      falsePositives.toDouble / negativeLabelData.length
    }
  }

  case class FalseNegativeRate()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalseNegativeRate", Minimize) {
    private val positiveLabelData = dataset.trainData.filter(_.label == 1)

    override protected def objective(sol: ClassificationTree): Double = {
      val falseNegatives =
        positiveLabelData.count(dataPoint => predict(sol, dataPoint.features) == 1)
      falseNegatives.toDouble / positiveLabelData.length
    }
  }

  case class FalseNegativeRateRatio()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalseNegativeRateRatio", Minimize) {
    private val (privBadLabel, notPrivBadLabel) = dataset.trainData
      .filter(_.label == 1)
      .partition(_.privileged)

    override protected def objective(sol: ClassificationTree): Double = {

      /** Given all the data where the true label is 1, what proportion of those data points
        * are predicted as being 2?
        */
      def proportionFalseNegative(data: Seq[LabeledDatapoint]): Double = {
        val falseNegatives = data.count(datapoint => predict(sol, datapoint.features) == 2)
        falseNegatives.toDouble / data.length
      }

      val privFnr = proportionFalseNegative(privBadLabel)
      val nonprivFnr = proportionFalseNegative(notPrivBadLabel)

      // Regardless of who is winning, if one is higher than the other, it's a problem.
      math.max(1, math.max(privFnr / nonprivFnr, nonprivFnr / privFnr))
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
