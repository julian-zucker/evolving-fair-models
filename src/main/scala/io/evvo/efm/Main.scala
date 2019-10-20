package io.evvo.efm

import java.nio.file.{Files, Paths}

import io.evvo.builtin.deletors.DeleteDominated
import io.evvo.builtin.trees.{ChangeLeafDataModifier, ChangeNodeDataModifier, SwapSubtreeModifier}
import io.evvo.efm.agents.{FullTreeCreator, LeafToNodeModifier}
import io.evvo.efm.ctree._
import io.evvo.efm.data.DataSet
import io.evvo.efm.objectives._
import io.evvo.island._
import io.evvo.island.population.Objective

import scala.concurrent.duration._
import scala.util.Using
import scala.util.chaining._

object Main {
  def main(args: Array[String]) {
    implicit val dataset: DataSet = DataSet.loadAdultIncome()

    // The list of fairness definitions to use.
    val fairnessesList: Seq[Seq[Objective[ClassificationTree]]] =
      Seq(
        Seq(TruePositiveRateRatio(), FalseNegativeRateRatio()),
        Seq(TruePositiveRateRatio(), DisparateImpact()),
        Seq(TruePositiveRateRatio(), BetweenGroupTheilIndex()),
        Seq(FalseNegativeRateRatio(), DisparateImpact()),
        Seq(FalseNegativeRateRatio(), BetweenGroupTheilIndex()),
        Seq(BetweenGroupTheilIndex(), DisparateImpact())
      )

    fairnessesList.foreach(fairnesses => {
      val islandBuilder = EvvoIslandBuilder()
        .addCreator(FullTreeCreator(depth = 5))
        .addModifier(ChangeLeafDataModifier[Decision, Label](_ => dataset.randomLabel()))
        .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeThreshold()))
        .addModifier(ChangeNodeDataModifier[Decision, Label](_.changeFeature()))
        .addModifier(LeafToNodeModifier())
        .addModifier(SwapSubtreeModifier())
        .withEmigrationStrategy(RandomSampleEmigrationStrategy(32, 10.seconds))
        .addDeletor(DeleteDominated())
        .addObjective(FalseNegativeRate())
        .addObjective(FalsePositiveRate())
        // Add each fairness metric in fairnesses to the builder
        .pipe(builder => fairnesses.foldRight(builder)((f, b) => b.addObjective(f)))

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

      val filePath = fairnesses match {
        case Nil => "results/data/test.csv"
        case fairness :: Nil =>
          f"results/data/one_fairness_metric/${fairness.name}|${dataset.name}.csv"
        case fairness1 :: fairness2 :: Nil =>
          f"results/data/two_fairness_metrics/${fairness1.name}|${fairness2.name}|${dataset.name}.csv"
      }

      Using(Files.newBufferedWriter(Paths.get(filePath))) { file =>
        file.write(table.split("\n").take(4).mkString("\n"))
        file.write("\n")
        file.write(results)
      }
    })
  }
}
