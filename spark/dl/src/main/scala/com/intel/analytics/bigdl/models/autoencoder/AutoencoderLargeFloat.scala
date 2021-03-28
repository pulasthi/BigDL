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

object AutoencoderLargeFloat {
  //  val rowN = 28
  //  val colN = 28
  //  val featureSize = rowN * colN
  val l2: Int = 756
  val l3: Int = 512
  val l4: Int = 256
  def apply(inputSize: Int): Module[Float] = {
    val model = Sequential[Float]()
    model.add(new Reshape(Array(inputSize)))
    model.add(new Linear(inputSize, l2))
    model.add(new ReLU[Float]())
    model.add(new Linear(l2, l3))
    model.add(new ReLU[Float]())
    model.add(new Linear(l3, l4))
    model.add(new ReLU[Float]())
    model.add(new Linear(l4, l3))
    model.add(new ReLU[Float]())
    model.add(new Linear(l3, l2))
    model.add(new ReLU[Float]())
    model.add(new Linear(l2, inputSize))
    model.add(new ReLU[Float]())
    model
  }
}
