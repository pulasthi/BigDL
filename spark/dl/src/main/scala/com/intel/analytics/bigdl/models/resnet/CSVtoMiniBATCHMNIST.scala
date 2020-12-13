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

package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

object CSVtoMiniBATCHMNIST {
  def apply(batchSize : Int) : CSVtoMiniBATCHMNIST = {
    new CSVtoMiniBATCHMNIST(batchSize)
  }
}

class CSVtoMiniBATCHMNIST(batchSize: Int)
  extends Transformer[Array[Double], MiniBatch[Double]] {


  override def apply(prev: Iterator[Array[Double]]): Iterator[MiniBatch[Double]] = {

    new Iterator[MiniBatch[Double]] {
      private val featureTensor: Tensor[Double] = Tensor[Double]()
      private val featureLabelTensor: Tensor[Double] = Tensor[Double]()
      private var featureData: Array[Double] = null
      private var featureLabel: Array[Double] = null


      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Double] = {
        if (prev.hasNext) {
          var i = 0
          var length = 0
          while (i < batchSize && prev.hasNext) {
            val arr = prev.next()
            length = arr.length
            if (featureData == null) {
              featureData = new Array[Double](batchSize * (arr.length - 1))
              featureLabel = new Array[Double](batchSize)
            }
            Array.copy(arr, 1, featureData, i * (arr.length - 1), arr.size - 1)
            featureLabel(i) = arr(0) + 1.0
            i += 1
          }
          featureTensor.set(Storage[Double](featureData),
            storageOffset = 1, sizes = Array(i, 1, 28, 28))
          featureLabelTensor.set(Storage[Double](featureLabel),
            storageOffset = 1, sizes = Array(i, 1))
          MiniBatch(featureTensor, featureLabelTensor)
        } else {
          null
        }
      }
    }
  }
}

