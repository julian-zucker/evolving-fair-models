package io.evvo

import io.evvo.ctree._
import io.evvo.data.DataSet
import io.evvo.island.population.{Maximize, Objective}

/** Holds objectives for ctrees. */
object objectives {

  case class Accuracy()(implicit dataset: DataSet)
    extends Objective[ClassificationTree]("Accuracy", Maximize) {
    override protected def objective(sol: ClassificationTree): Double = {
      val correct = dataset.data.map(
        dataPoint => dataPoint.label == predict(sol, dataPoint.features))
      correct.count(identity).toDouble / correct.length
    }
  }
}
