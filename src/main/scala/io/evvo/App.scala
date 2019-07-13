package io.evvo

import io.evvo.agent.defaults.trees.ChangeLeafDataModifier
import io.evvo.island.population.{Minimize, Objective}


object App {

  val mod = ChangeLeafDataModifier[Boolean, Int](_ + 1)


  class TestObj extends Objective[Double]("test", Minimize) {
    override protected def objective(sol: Double): Double = {
      3
    }
  }

  def main(args : Array[String]) {
    println( "Hello World!" )
  }

}
