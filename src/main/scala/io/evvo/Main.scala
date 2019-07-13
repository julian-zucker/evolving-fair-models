package io.evvo

import io.evvo.data.DataSet

object Main {
  def main(args: Array[String]) {
    val german = DataSet.load("german")
    println(german)
    println(german.data.take(10))
  }
}
