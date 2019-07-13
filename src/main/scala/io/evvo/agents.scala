package io.evvo

import io.evvo.agent.defaults.trees.{BTLeaf, BTNode}
import io.evvo.agent.{CrossoverFunction, MutatorFunction}
import io.evvo.ctree.{ClassificationTree, Decision, Label}
import io.evvo.data.DataSet

object agents {

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
