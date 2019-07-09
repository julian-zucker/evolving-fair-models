package io.evvo

import io.evvo.island.population.{Minimize, Objective}


object App {

  class TestObj extends Objective[Double]("test", Minimize) {
    override protected def objective(sol: Double): Double = {
      3
    }
  }

  def main(args : Array[String]) {
    println( "Hello World!" )
  }

}
