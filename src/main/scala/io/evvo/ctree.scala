package io.evvo

import io.evvo.data.DataPoint
import io.evvo.agent.defaults.trees.{BTLeaf, BTNode, BinaryTree}

/**
  * `ctree` is short for classification tree. This object holds data representations and
  * utility functions for `ctree`s.
  */
object ctree {
  /** A tree representing a decision tree on continuous variables with categorical output. */
  type ClassificationTree = BinaryTree[Decision, Label]

  /**
    * Represents a decision point.
    * @param label The label to make a decision based on
    * @param threshold The threshold: if the label value for this datapoint is lower than the
    *                  threshold, go left, otherwise go right.
    */
  case class Decision(label: Label, threshold: Float) {
    /** Picks left if the datapoint's label is less than threshold, right otherwise. */
    def pick[T](dataPoint: DataPoint, left: T, right: T): T = {
      if (dataPoint(label.index) < threshold) left else right
    }
  }

  /** Represents a label (by index). */
  case class Label(index: Int)

  /** What label does the given classification tree predict on the given datapoint? */
  def predict(classificationTree: ClassificationTree, datapoint: DataPoint): Label = {
    classificationTree match {
      case BTLeaf(label) => label
      case BTNode(decision, left, right) =>
        predict(decision.pick(datapoint, left, right), datapoint)
    }
  }
}
