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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
object ConvolutionMN {
  def apply[@specialized(Float, Double) T: ClassTag]
  ( nInputPlane: Int,
    nOutputPlane: Int,
    kernelW: Int,
    kernelH: Int,
    strideW: Int = 1,
    strideH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    nGroup: Int = 1,
    propagateBack: Boolean = true,
    optnet: Boolean = false,
    weightDecay: Double = 1e-4)
  (implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    val wReg = L2Regularizer[T](weightDecay)
    val bReg = L2Regularizer[T](weightDecay)
    val conv = if (optnet) {
      SpatialShareConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    } else {
      SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    }
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object CNNmodel {

  val featureSize = 3*3*64

  def apply(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
    model.add(ConvolutionMN(1, 32, 5, 5))
    model.add(new ReLU[Float]())
    model.add(ConvolutionMN(32, 32, 5, 5))
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(new ReLU[Float]())
    model.add(new Dropout[Float](0.5))
    model.add(ConvolutionMN(32, 64, 5, 5))
    model.add(SpatialMaxPooling(2, 2, 2, 2))
    model.add(new ReLU[Float]())
    model.add(new Dropout[Float](0.5))
    model.add(new Reshape[Float](Array(featureSize)))
    model.add(new Linear(featureSize, 256))
    model.add(new ReLU[Float]())
    model.add(new Dropout[Float](0.5))
    model.add(new Linear(256, 10))
    model.add(new LogSoftMax[Float]())
    model
  }
}

