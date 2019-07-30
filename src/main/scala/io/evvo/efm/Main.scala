package io.evvo.efm

import java.nio.file.{Files, Paths}

import io.evvo.builtin.deletors.{DeleteDominated, DeleteWorstHalfByRandomObjective}
import io.evvo.builtin.trees.{ChangeLeafDataModifier, ChangeNodeDataModifier, SwapSubtreeModifier}
import io.evvo.efm.agents.{FullTreeCreator, LeafToNodeModifier}
import io.evvo.efm.ctree._
import io.evvo.efm.data.DataSet
import io.evvo.efm.objectives.{BetweenGroupTheilIndex, DisparateImpact, FalseNegativeRate, FalseNegativeRateRatio, FalsePositiveRate}
import io.evvo.island._

import scala.concurrent.duration._
import scala.util.Using

object Main {
  def main(args: Array[String]) {
    implicit val dataset: DataSet = DataSet.load("COMPAS")

    val fairness = BetweenGroupTheilIndex()

    val islandBuilder = EvvoIslandBuilder()
//      .addCreator(LeafCreator[Decision, Label](Seq(() => true, () => false)))
      .addCreator(FullTreeCreator(depth = 5))
      .addModifier(ChangeLeafDataModifier[Decision, Label](_ => dataset.randomLabel()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeThreshold()))
      .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeFeature()))
      .addModifier(LeafToNodeModifier())
      .addModifier(SwapSubtreeModifier())
      .addDeletor(DeleteDominated())
      .addObjective(FalseNegativeRate())
      .addObjective(FalsePositiveRate())
      .addObjective(fairness)
      .withEmigrationStrategy(RandomSampleEmigrationStrategy(32, 10.seconds))
      .withLoggingStrategy(LogPopulation(durationBetweenLogs = 1.minute))

    val islandManager = new LocalIslandManager(4, islandBuilder)

    islandManager.runBlocking(StopAfter(5.minutes))

    val table = islandManager.currentParetoFrontier().toTable()
    val trees = islandManager
      .currentParetoFrontier()
      .solutions
      .toVector
      .sortBy(_.scoreOn("FalseNegativeRate"))

    val preds = trees.map(tree => dataset.testData.map(_.predictedCorrectlyBy(tree.solution)))
    val accuracies = preds.map(p => p.count(identity).toDouble / p.length)
    val results =
      table
        .split("\n")
        .drop(4)
        .zip(accuracies)
        .map {
          case (data, acc) => f"${data.split("\t").mkString(",")},${acc}"
        }
        .mkString("\n")

    Using(Files.newBufferedWriter(Paths.get(f"results/data/${fairness.name}|${dataset.name}.csv"))) {
      _.write(results)
    }
  }
}
