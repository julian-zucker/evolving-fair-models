package io.evvo.efm

import io.evvo.builtin.trees.{BTLeaf, BTNode, BinaryTree}
import io.evvo.efm.data.{DataPoint, DataSet}

/**
  * `ctree` is short for classification tree. This object holds data representations and
  * utility functions for `ctree`s.
  */
object ctree {

  /** A tree representing a decision tree on continuous variables with categorical output. */
  type ClassificationTree = BinaryTree[Decision, Label]

  /**
    * Represents a decision point.
    *
    * @param featureIndex The label to make a decision based on
    * @param threshold    The threshold: if the label value for this datapoint is lower than the
    *                     threshold, go left, otherwise go right.
    */
  case class Decision(featureIndex: Int, threshold: Double)(implicit val dataset: DataSet) {

    /** @return `left` if the datapoint's label is less than threshold, `right` otherwise. */
    def pick[T](dataPoint: DataPoint, left: T, right: T): T = {
      if (dataPoint(featureIndex) < threshold) left else right
    }

    /** @return This Decision, but using a new threshold to split */
    def changeThreshold(): Decision = {
      // Pick randomly between the min and max values found in the feature
      val featureValues = dataset.featureValues(featureIndex)
      val scale = featureValues.max - featureValues.min
      val min = featureValues.min

      this.copy(threshold = util.Random.nextDouble() * scale + min)
    }

    /** @return This Decision, but splitting on a different feature. */
    def changeFeature(): Decision = {
      this.copy(featureIndex = util.Random.nextInt(dataset.numFeatures))
    }
  }

  object Decision {

    /** @return a random decision, on any feature, with a split somewhere in the middle. */
    def randomDecision()(implicit dataSet: DataSet): Decision = {
      val feature = util.Random.nextInt(dataSet.numFeatures)
      val threshold = dataSet.featureValues(feature)(util.Random.nextInt(dataSet.trainData.length))
      Decision(feature, threshold)
    }
  }

  /** Represents a label (by index). */
  sealed trait Label
  case object Positive extends Label
  case object Negative extends Label

  /** What label does the given classification tree predict on the given datapoint? */
  def predict(classificationTree: ClassificationTree, datapoint: DataPoint): Label = {
    classificationTree match {
      case BTLeaf(label) => label
      case BTNode(decision, left, right) =>
        predict(decision.pick(datapoint, left, right), datapoint)
    }
  }
}
