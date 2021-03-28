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
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.text.CSVtoMiniBatchFloat
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Transformer}
import com.intel.analytics.bigdl.nn.{MSECriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.{Engine, OptimizerV1, OptimizerV2, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.Array.ofDim
import scala.io.Source
import scala.reflect.ClassTag

object toAutoencoderLargeFloatBatch {
  def apply(): toAutoencoderLargeFloatBatch[Float] = new toAutoencoderLargeFloatBatch[Float]()
}

class toAutoencoderLargeFloatBatch[T: ClassTag](implicit ev: TensorNumeric[T]
      )extends Transformer[MiniBatch[T], MiniBatch[T]] {
  override def apply(prev: Iterator[MiniBatch[T]]): Iterator[MiniBatch[T]] = {
    prev.map(batch => {
      MiniBatch(batch.getInput().toTensor[T], batch.getInput().toTensor[T])
    })
  }
}

object TrainLargeFloat {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)


  import Utils._

  def loadCSV(csvPath: String): Array[Array[Float]] = {
    Source.fromFile(csvPath)
      .getLines()
      .map(_.split(",").map(_.trim.toFloat))
      .toArray
  }

  def  genData(dataSize: Int, inputSize: Int) : Array[Array[Float]] = {
    val data: Array[Array[Float]] = ofDim[Float](dataSize, inputSize);
    val random = scala.util.Random
    for (i <- 0 until data.length) {
      for ( j <- 0 until data(0).length) {
        data(i)(j) = random.nextFloat();
      }
    }

    data
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      val startTime = System.nanoTime()
      val dataLoadTime = System.nanoTime()

      val conf = Engine.createSparkConf().setAppName("Train AutoencoderLarge on MNIST")
        .set("spark.rpc.message.maxSize", "512")
      val sc = new SparkContext(conf)
      Engine.init

//      val trainData = Paths.get(param.folder, "/train-images-idx3-ubyte")
//      val trainLabel = Paths.get(param.folder, "/train-labels-idx1-ubyte")

//      val trainDataSet = DataSet.array(loadCSV(param.folder), sc) ->
//        CSVtoMiniBatchFloat(param.batchSize) -> toAutoencoderLargeFloatBatch()

      val trainDataSet = DataSet.array(genData(param.dataSize, param.inputSize), sc) ->
              CSVtoMiniBatchFloat(param.batchSize) -> toAutoencoderLargeFloatBatch()
      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        AutoencoderLargeFloat(inputSize = param.inputSize, numLayers = param.layers)
      }
      if (param.optimizerVersion.isDefined) {
        param.optimizerVersion.get.toLowerCase match {
          case "optimizerv1" => Engine.setOptimizerVersion(OptimizerV1)
          case "optimizerv2" => Engine.setOptimizerVersion(OptimizerV2)
        }
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adagrad[Float](learningRate = 0.01, learningRateDecay = 0.0, weightDecay = 0.0005)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new MSECriterion[Float]()
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      optimizer
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
      sc.stop()
    })
  }
}

