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

object AutoencoderLarge {
  val rowN = 28
  val colN = 28
  val featureSize = 100

  def apply(classNum: Int): Module[Double] = {
    val model = Sequential[Double]()
    val l1 = 1024
    val l2 = 512
    val l3 = 128
    val l4 = 8
    model.add(new Reshape(Array(l1)))
    model.add(new Linear(l1, l2))
    model.add(new ReLU[Double]())
//    model.add(new Linear(l2, l3))
//    model.add(new ReLU[Double]())
//    model.add(new Linear(l3, l4))
//    model.add(new ReLU[Double]())
//    model.add(new Linear(l4, l3))
//    model.add(new ReLU[Double]())
//    model.add(new Linear(l3, l2))
//    model.add(new ReLU[Double]())
    model.add(new Linear(l2, l1))
    model.add(new Sigmoid[Double]())
    model
  }

  def graph(classNum: Int): Module[Double] = {
    val input = Reshape[Double](Array(featureSize)).inputs()
    val linear1 = Linear[Double](featureSize, classNum).inputs(input)
    val relu = ReLU[Double]().inputs(linear1)
    val linear2 = Linear[Double](classNum, featureSize).inputs(relu)
    val output = Sigmoid[Double]().inputs(linear2)
    Graph[Double](input, output)
  }
}
