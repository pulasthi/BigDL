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

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.text.CSVtoMiniBatch
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, Transformer}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, MSECriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.{Engine, OptimizerV1, OptimizerV2, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.reflect.ClassTag
import scala.io.Source

object toMnistBatch {
  def apply(): toMnistBatch[Float] = new toMnistBatch[Float]()
}

class toMnistBatch[T: ClassTag](implicit ev: TensorNumeric[T]
      )extends Transformer[MiniBatch[T], MiniBatch[T]] {
  override def apply(prev: Iterator[MiniBatch[T]]): Iterator[MiniBatch[T]] = {
    prev.map(batch => {
      MiniBatch(batch.getInput().toTensor[T], batch.getTarget().toTensor[T])
    })
  }
}

object TrainMNIST {
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

  def arrayToSample(testDataArray: Array[Array[Float]]): (Array[Sample[Float]], Array[Int]) = {

    val data = new Array[Sample[Float]](testDataArray.length)
    val labels = new Array[Int](testDataArray.length)
    var i = 0;
    var featureData: Array[Float] = null
    var featureLabel: Array[Float] = null


    while (i < testDataArray.length) {
      featureData = new Array[Float](testDataArray(i).length - 1)
      featureLabel = new Array[Float](1)
      Array.copy(testDataArray(i), 1, featureData, 0, testDataArray(i).size - 1)
      featureLabel(0) = testDataArray(i)(0) + 1.0f
      labels(i) = testDataArray(i)(0).toInt + 1
      val featureTensor: Tensor[Float] = Tensor[Float]()
      val featureLabelTensor: Tensor[Float] = Tensor[Float]()
      featureTensor.set(Storage[Float](featureData),
        storageOffset = 1, sizes = Array(1, 28, 28))
      featureLabelTensor.set(Storage[Float](featureLabel),
        storageOffset = 1, sizes = Array(1))

      data(i) = Sample(featureTensor, featureLabelTensor)
      i += 1
    }


    (data, labels)
  }

  def main(args: Array[String]): Unit = {
    trainParserMnist.parse(args, new TrainParamsMNist()).map(param => {
      val startTime = System.nanoTime()

      val conf = Engine.createSparkConf().setAppName("Train Autoencoder on MNIST")
        .set("spark.rpc.message.maxSize", "512")
      val sc = new SparkContext(conf)
      Engine.init

      val trainDataPath = Paths.get(param.folder, "/train640.csv")
      val testDataPath = Paths.get(param.folder, "/mnist_test_small.csv")

      val trainDataSet = DataSet.array(loadCSV(trainDataPath.toString), sc) ->
        CSVtoMiniBATCHMNIST(param.batchSize)

      val testDataArray : Array[Array[Float]] = loadCSV(testDataPath.toString)
      var (testDataSet, labels) = arrayToSample(testDataArray)
      val testDataSetRDD = sc.parallelize(testDataSet, 1)

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        CNNmodel(classNum = 12)
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
        new Adam[Float]()
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      optimizer
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
      val endTime = System.nanoTime()

//      var result = model.predictClass(testDataSetRDD, 1).collect()
//      var errorCount = 0
//      for ( i <- 0 until result.length ) {
//        if ( result(i) != labels(i) ) {
//          errorCount += 1
//        }
//      }
//
//      var precent = 1.0 - errorCount.toFloat/result.length
//      println("Test Accuracy : " + precent*100 + "%")
//      print("Total Time : " + (endTime - startTime) / 1000000 + "ms")

      sc.stop()


    })
  }
}

