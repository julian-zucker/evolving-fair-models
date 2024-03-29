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
    private val negativeLabelData = dataset.trainData.filter(_.label == Negative)

    override protected def objective(sol: ClassificationTree): Double = {
      val falsePositives = negativeLabelData.count(!_.predictedCorrectlyBy(sol))
      falsePositives.toDouble / dataset.trainData.length
    }
  }

  case class FalseNegativeRate()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FalseNegativeRate", Minimize) {
    private val positiveLabelData = dataset.trainData.filter(_.label == Positive)

    override protected def objective(sol: ClassificationTree): Double = {
      val falseNegatives = positiveLabelData.count(!_.predictedCorrectlyBy(sol))
      falseNegatives.toDouble / dataset.trainData.length
    }
  }

  // ===============================================================================================
  // Fairness Metrics
  /** Measures the disparity between false negative rates in the two groups. */
  case class FalseNegativeRateRatio()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("FNR Ratio", Minimize) {
    private val (privPosLabel, unprivPosLabel) = dataset.trainData
      .filter(_.label == Positive)
      .partition(_.isPrivileged)

    private val trainSize = dataset.trainData.length

    override protected def objective(sol: ClassificationTree): Double = {
      def proportionFalseNegative(data: Seq[LabeledDatapoint]): Double = {
        data.count(!_.predictedCorrectlyBy(sol)).toDouble / data.length
      }
      val privFalseNegRate = proportionFalseNegative(privPosLabel)
      val nonprivFalsePosRate = proportionFalseNegative(unprivPosLabel)

      ratio(privFalseNegRate, nonprivFalsePosRate)
    }

    override def toString: String = "FNR Ratio"
  }

  /** Measures the ratio at which the privileged and non-privileged groups are given positive
    * labels.
    */
  case class DisparateImpact()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("Disparate Impact", Minimize) {
    private val (privilegedGroup, unprivilegedGroup) =
      dataset.trainData.partition(_.isPrivileged)

    override protected def objective(sol: ClassificationTree): Double = {
      def proportionPositiveLabel(data: Seq[LabeledDatapoint]): Double = {
        data.count(_.predictionFrom(sol) == Positive).toDouble / data.length
      }
      val ppRatio = proportionPositiveLabel(privilegedGroup)
      val nppRatio = proportionPositiveLabel(unprivilegedGroup)

      ratio(ppRatio, nppRatio)
    }
  }

  /** Measures the disparity between true positive rates in the two groups. */
  case class TruePositiveRateRatio()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("TPR Ratio", Minimize) {
    private val (privilegedGroup, notPrivilegedGroup) = dataset.trainData
      .filter(_.label == Positive)
      .partition(_.isPrivileged)

    override protected def objective(sol: ClassificationTree): Double = {
      def proportionAccurate(data: Seq[LabeledDatapoint]): Double = {
        data.count(_.predictedCorrectlyBy(sol)).toDouble / data.length
      }
      // Accurate within the positive labeled data == true positive.
      val privTruePos = proportionAccurate(privilegedGroup)
      val unprivTruePos = proportionAccurate(notPrivilegedGroup)

      ratio(privTruePos, unprivTruePos)
    }
  }

  case class BetweenGroupTheilIndex()(implicit dataset: DataSet)
      extends Objective[ClassificationTree]("Between-Group Theil Index", Minimize, 5) {
    // implementation derived from https://github.com/IBM/AIF360/blob/d499b4ad1d3557866d6807cfb2dfbdfe4b6ba361/aif360/metrics/classification_metric.py#L650
    override protected def objective(sol: ClassificationTree): Double = {
      val (priv, unpriv) = dataset.trainData.partition(_.isPrivileged)

      val privBenefit = mean(benefits(sol, priv))
      val unprivBenefit = mean(benefits(sol, unpriv))
      // Theil index of the mean benefits to each group, weighted by group size
      theilIndex(Seq.fill(priv.length)(privBenefit) ++ Seq.fill(unpriv.length)(unprivBenefit))
    }

    private def theilIndex(benefits: Seq[Double]): Double = {
      val meanBenefit = mean(benefits)
      val log = benefits.map(x => math.log(math.pow(x / meanBenefit, x)) / meanBenefit)
      mean(log)
    }

    private def benefits(tree: ClassificationTree, data: Seq[LabeledDatapoint]): Seq[Double] = {
      val predLabels = data.map(_.predictionFrom(tree)).map(labelToDouble)
      val trueLabels = data.map(_.label).map(labelToDouble)

      // Corresponds to 1, 1, 2, 0 in the last row of Figure 2 in the paper
      // "A Unified Approach to Quantifying Algorithmic Unfairness:
      //    Measuring Individual and Group Unfairness via Inequality Indices", Speicher et al.
      predLabels.zip(trueLabels).map { case (p, t) => 1d + p - t }
    }

    private def mean(s: Seq[Double]): Double = s.sum / s.length

    private def labelToDouble(label: Label): Double = if (label == Positive) { 1d } else { 0d }

    override def toString: String = "TheilIndex"
  }

  /** @return The higher of `n1/n2` and `n2/n1`. If n1 and n2 are both zero, returns 1.
    *         If only one of n1 and n2 is 0, returns infinity.
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
