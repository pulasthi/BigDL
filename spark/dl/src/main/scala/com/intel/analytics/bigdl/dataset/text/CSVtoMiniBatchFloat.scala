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

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

object CSVtoMiniBatchFloat {
  def apply(batchSize : Int) : CSVtoMiniBatchFloat = {
    new CSVtoMiniBatchFloat(batchSize)
  }
}

class CSVtoMiniBatchFloat private[dataset](batchSize: Int)
  extends Transformer[Array[Float], MiniBatch[Float]] {


  override def apply(prev: Iterator[Array[Float]]): Iterator[MiniBatch[Float]] = {

    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null


      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          var i = 0
          var length = 0
          while (i < batchSize && prev.hasNext) {
            val arr = prev.next()
            length = arr.length
            if (featureData == null) {
              featureData = new Array[Float](batchSize * arr.length)
            }
            Array.copy(arr, 0, featureData, i * arr.length, arr.size)
            i += 1
          }
          featureTensor.set(Storage[Float](featureData),
            storageOffset = 1, sizes = Array(i, 1, length))
          MiniBatch(featureTensor, null)
        } else {
          null
        }
      }
    }
  }
}