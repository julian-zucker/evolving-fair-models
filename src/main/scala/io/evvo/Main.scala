package io.evvo

import io.evvo.agent.defaults.DeleteDominated
import io.evvo.agent.defaults.trees._
import io.evvo.agents.FullTreeCreator
import io.evvo.ctree.{Decision, Label}
import io.evvo.data.DataSet
import io.evvo.island._
import io.evvo.objectives.{FalseNegativeRate, FalsePositiveRate}

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

    val islandManager = new LocalIslandManager(4, islandBuilder)

    islandManager.runBlocking(StopAfter(10.second))

    println(islandManager.currentParetoFrontier())
    println(islandManager.currentParetoFrontier().solutions.take(5))

  }
}
