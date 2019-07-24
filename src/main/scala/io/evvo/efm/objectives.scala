package io.evvo.efm

import io.evvo.efm.ctree._
import io.evvo.efm.data.{DataSet, LabeledDatapoint}
import io.evvo.island.population.{Maximize, Minimize, Objective}

/** Holds objectives for ctrees. */
object objectives {

  case class Accuracy()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("Accuracy", Maximize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val correctCount = dataset.trainData.count(_.predictedCorrectlyBy(sol))
      correctCount.toDouble / dataset.trainData.length
    }
  }

  case class FalsePositiveRate()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalsePositiveRate", Minimize) {
    private val negativeLabelData = dataset.trainData.filter(!_.positiveLabel)

    override protected def objective(sol: ClassificationTree): Double = {
      val falsePositives = negativeLabelData.count(!_.predictedCorrectlyBy(sol))
      falsePositives.toDouble / dataset.trainData.length
    }
  }

  case class FalseNegativeRate()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalseNegativeRate", Minimize) {
    private val positiveLabelData = dataset.trainData.filter(_.positiveLabel)

    override protected def objective(sol: ClassificationTree): Double = {
      val falseNegatives = positiveLabelData.count(!_.predictedCorrectlyBy(sol))
      falseNegatives.toDouble / dataset.trainData.length
    }
  }

  case class FalseNegativeRateRatio()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalseNegativeRateRatio", Minimize) {
    private val (privPosLabel, notPrivPosLabel) = dataset.trainData
      .filter(_.positiveLabel)
      .partition(_.privileged)

    private val trainSize = dataset.trainData.length

    override protected def objective(sol: ClassificationTree): Double = {
      def proportionFalseNegative(data: Seq[LabeledDatapoint]): Double = {
        data.count(!_.predictedCorrectlyBy(sol)).toDouble / data.length
      }
      val privFalseNegRate = proportionFalseNegative(privPosLabel)
      val nonprivFalsePosRate = proportionFalseNegative(notPrivPosLabel)

      ratio(privFalseNegRate, nonprivFalsePosRate)
    }
  }

  /** Measures the ratio at which the privileged and non-privileged groups are given label 1. */
  case class DisparateImpact()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("DisparateImpact", Minimize) {
    private val (priviligedGroup, notPriviligedGroup) = dataset.trainData
      .partition(_.privileged)

    override protected def objective(sol: ClassificationTree): Double = {
      def proportionAccurate(data: Seq[LabeledDatapoint]): Double = {
        data.count(_.predictedCorrectlyBy(sol)).toDouble / data.length
      }
      val ppRatio = proportionAccurate(priviligedGroup)
      val nppRatio = proportionAccurate(notPriviligedGroup)

      ratio(ppRatio, nppRatio)
    }
  }

  /** @return The higher of `n1/n2` and `n2/n1`. If n1 and n2 are both zero, returns 1. If
    *         only one of n1 and n2 is 0, returns infinity.
    */
  def ratio(n1: Double, n2: Double): Double = {
    // Regardless of who is winning, if one is higher than the other, it's a problem,
    // so ensure it's always above one.
    if (n1 == 0 && n2 == 0) {
      // If both are zero, the ratio is 1, if one is zero, it's caught the try block.
      1
    } else {
      try {
        math.max(n1 / n2, n2 / n1)
      } catch {
        // If one has 0 FNR, and the other doesn't, the ratio is infinity
        case e: ArithmeticException => Double.PositiveInfinity
      }
    }
  }
}
