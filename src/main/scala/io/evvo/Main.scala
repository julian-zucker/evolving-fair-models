package io.evvo

import io.evvo.agent.defaults.DeleteDominated
import io.evvo.agent.defaults.trees.{ChangeLeafDataModifier, ChangeNodeDataModifier, LeafCreator, SwapSubtreeModifier}
import io.evvo.agents.LeafToNodeModifier
import io.evvo.ctree.{Decision, Label}
import io.evvo.data.DataSet
import io.evvo.island._
import io.evvo.objectives.Accuracy

import scala.concurrent.duration._

object Main {
  def main(args: Array[String]) {
    implicit val dataset: DataSet = DataSet.load("german")

    val islandBuilder = EvvoIslandBuilder()
      // Create leaves with each of the possible leaf values
      .addCreator(LeafCreator[Decision, Label](dataset.possibleLabels.map(() => _)))
      // Change leaves to random labels
      .addModifier(ChangeLeafDataModifier[Decision, Label](_ => dataset.randomLabel()))
      // Change the threshold of a node
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeThreshold()))
      // Change the feature that a node runs on
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeFeature()))
      // Swap a leaf out for a node
      .addModifier(LeafToNodeModifier())
      .addModifier(SwapSubtreeModifier())
      .addDeletor(DeleteDominated())
      .addObjective(Accuracy())

    val islandManager = new LocalIslandManager(10, islandBuilder)

    islandManager.runBlocking(StopAfter(10.second))
    println(islandManager.currentParetoFrontier().solutions)
  }
}
