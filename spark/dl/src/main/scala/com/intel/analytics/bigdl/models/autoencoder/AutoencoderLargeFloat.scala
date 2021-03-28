/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.autoencoder

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Graph, _}

import scala.Array.ofDim

object AutoencoderLargeFloat {

  def apply(inputSize: Int, numLayers: Int): Module[Float] = {
    val model = Sequential[Float]()
    model.add(new Reshape(Array(inputSize)))

    val layers: Array[Int] = new Array[Int](numLayers + 1);
    layers(0) = inputSize;
    for ( i <- 1 until layers.length) {
      layers(i) = (layers(i - 1) * .75).toInt
    }

    println("############  :  " + layers.mkString(","))

    for ( f <- 1 until layers.length) {
      model.add(new Linear(layers(f - 1), layers(f)))
      model.add(new ReLU[Float]())
    }

    for ( b <- (layers.length - 1)  to 1 by -1) {
      model.add(new Linear(layers(b), layers(b - 1)))
      model.add(new ReLU[Float]())
    }

    model
  }
}
