package io.evvo.efm

import io.evvo.builtin.deletors.{DeleteDominated, DeleteWorstHalfByRandomObjective}
import io.evvo.builtin.trees.{ChangeLeafDataModifier, ChangeNodeDataModifier, SwapSubtreeModifier}
import io.evvo.efm.agents.{FullTreeCreator, LeafToNodeModifier}
import io.evvo.efm.ctree._
import io.evvo.efm.data.DataSet
import io.evvo.efm.objectives.{FalseNegativeRate, FalsePositiveRate, TruePositiveRateRatio}
import io.evvo.island._

import scala.concurrent.duration._

object Main {
  def main(args: Array[String]) {
    implicit val dataset: DataSet = DataSet.load("compas")
    println(dataset)

    val islandBuilder = EvvoIslandBuilder()
//      .addCreator(LeafCreator[Decision, Label](Seq(() => true, () => false)))
      .addCreator(FullTreeCreator(depth = 5))
      .addModifier(ChangeLeafDataModifier[Decision, Label](_ => dataset.randomLabel()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeThreshold()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeFeature()))
      .addModifier(LeafToNodeModifier())
      .addModifier(SwapSubtreeModifier())
      .addDeletor(DeleteDominated())
      .addDeletor(DeleteWorstHalfByRandomObjective(10))
      .addObjective(FalseNegativeRate())
      .addObjective(FalsePositiveRate())
      .addObjective(TruePositiveRateRatio())
//      .addObjective(DisparateImpact())
      .withEmigrationStrategy(RandomSampleEmigrationStrategy(32, 10.seconds))
      .withLoggingStrategy(LogPopulation(durationBetweenLogs = 10.second))

    val islandManager = new LocalIslandManager(5, islandBuilder)

    islandManager.runBlocking(StopAfter((5 * 60).second))

    val table = islandManager.currentParetoFrontier().toTable()
    val trees = islandManager
      .currentParetoFrontier()
      .solutions
      .toVector
      .sortBy(_.scoreOn("FalseNegativeRate"))

    val preds = trees.map(tree => dataset.testData.map(_.predictedCorrectlyBy(tree.solution)))
    val accuracies = preds.map(p => p.count(identity).toDouble / p.length)
    println(table.split("\n").take(4).mkString("\n"))
    println(
      table
        .split("\n")
        .drop(4)
        .zip(accuracies)
        .map {
          case (data, acc) => f"${data}\t${acc}"
        }
        .mkString("\n"))
  }
}

