package io.evvo.efm

import io.evvo.builtin.deletors.DeleteDominated
import io.evvo.builtin.trees.{ChangeLeafDataModifier, ChangeNodeDataModifier, LeafCreator, SwapSubtreeModifier}
import io.evvo.efm.agents.FullTreeCreator
import io.evvo.efm.ctree._
import io.evvo.efm.data.DataSet
import io.evvo.efm.objectives.{FalseNegativeRate, FalsePositiveRate}
import io.evvo.island.{EvvoIslandBuilder, LocalIslandManager, StopAfter}

import scala.concurrent.duration._

object Main {
  def main(args: Array[String]) {
    implicit val dataset: DataSet = DataSet.load("german")

    val islandBuilder = EvvoIslandBuilder()
      .addCreator(LeafCreator[Decision, Label](dataset.possibleLabels.map(() => _)))
      .addCreator(FullTreeCreator(depth = 5))
      .addModifier(ChangeLeafDataModifier[Decision, Label](_ => dataset.randomLabel()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeThreshold()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeFeature()))
      //      .addModifier(LeafToNodeModifier())
      .addModifier(SwapSubtreeModifier())
      .addDeletor(DeleteDominated())
      .addObjective(FalseNegativeRate())
      .addObjective(FalsePositiveRate())
    //      .addObjective(Accuracy())

    val islandManager = new LocalIslandManager(4, islandBuilder)

    islandManager.runBlocking(StopAfter(1.second))

    println(islandManager.currentParetoFrontier())
    val trees = islandManager.currentParetoFrontier().solutions

    val preds = trees.toVector.map(tree =>
      dataset.testData.map(x => x.label == ctree.predict(tree.solution, x.features)))
    val accuracies = preds.map(p => p.count(identity).toDouble / p.length)
    println(f"Accuracy: ${accuracies}")
  }
}
