package io.evvo.efm

import io.evvo.builtin.trees.{BTLeaf, BTNode, BinaryTree}
import io.evvo.agent.{CreatorFunction, MutatorFunction}
import io.evvo.builtin.trees
import io.evvo.efm.ctree.{ClassificationTree, Decision, Label}
import io.evvo.efm.data.DataSet

object agents {
  case class FullTreeCreator(depth: Int)(implicit dataSet: DataSet)
    extends CreatorFunction[BinaryTree[Decision, Label]]("FullTreeCreator") {
    override def create(): Iterable[BinaryTree[Decision, Label]] =
      Vector.fill(32)(makeOneTree(depth))

    def makeOneTree(depth: Label): BinaryTree[Decision, Label] = {
      if (depth == 0) {
        BTLeaf(dataSet.randomLabel())
      } else {
        trees.BTNode(Decision.randomDecision(), makeOneTree(depth - 1), makeOneTree(depth - 1))
      }
    }
  }

  case class LeafToNodeModifier()(implicit dataSet: DataSet)
    extends MutatorFunction[ClassificationTree]("LeafToNode") {
    override protected def mutate(sol: ClassificationTree): ClassificationTree = {
    sol match {
        case l: BTLeaf[Decision, Label] => BTNode(Decision.randomDecision(), l, l)
        case n: BTNode[Decision, Label] =>
          BTNode(Decision.randomDecision(), n, n)
      }
    }
  }
}
