# 实验四

- 使用[TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification)训练自定义的图像分类器
- 利用Android Studio导入训练后的模型，并结合CameraX使用
- 利用手机GPU加速模型运行

## 创建工作目录

```
git clone https://github.com/hoitab/TFLClassify.git
```

## 下载相关库

## 连接物理机运行start模块

## 选择已经下载的自定义的训练模型

> 模型训练任务以后完成，这里选择finish模块中ml文件下的FlowerModel.tflite

<img src="pic\1.png" width="50%" />

## 添加代码

```kotlin
 private class ImageAnalyzer(ctx: Context, private val listener: RecognitionListener) :
        ImageAnalysis.Analyzer {

        // TODO 1: Add class variable TensorFlow Lite Model
        private val flowerModel = FlowerModel.newInstance(ctx)
        // Initializing the flowerModel by lazy so that it runs in the same thread when the process
        // method is called.

        // TODO 6. Optional GPU acceleration


        override fun analyze(imageProxy: ImageProxy) {

            val items = mutableListOf<Recognition>()

            // TODO 2: Convert Image to Bitmap then to TensorImage
            val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
            // TODO 3: Process the image using the trained model, sort and pick out the top results
            val outputs = flowerModel.process(tfImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // Sort with highest confidence first
                }.take(MAX_RESULT_DISPLAY) // take the top results
            // TODO 4: Converting the top probability items into a list of recognitions
            for (output in outputs) {
                items.add(Recognition(output.label, output.score))
            }
            // START - Placeholder code at the start of the codelab. Comment this block of code out.
//            for (i in 0..MAX_RESULT_DISPLAY-1){
//                items.add(Recognition("Fake label $i", Random.nextFloat()))
//            }
            // END - Placeholder code at the start of the codelab. Comment this block of code out.


            // Return the result
            listener(items.toList())

            // Close the image,this tells CameraX to feed the next image to the analyzer
            imageProxy.close()
        }

        /**
         * Convert Image Proxy to Bitmap
         */
        private val yuvToRgbConverter = YuvToRgbConverter(ctx)
        private lateinit var bitmapBuffer: Bitmap
        private lateinit var rotationMatrix: Matrix

        @SuppressLint("UnsafeExperimentalUsageError")
        private fun toBitmap(imageProxy: ImageProxy): Bitmap? {

            val image = imageProxy.image ?: return null

            // Initialise Buffer
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                Log.d(TAG, "Initalise toBitmap()")
                rotationMatrix = Matrix()
                rotationMatrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }

            // Pass image to an image analyser
            yuvToRgbConverter.yuvToRgb(image, bitmapBuffer)

            // Create the Bitmap in the correct orientation
            return Bitmap.createBitmap(
                bitmapBuffer,
                0,
                0,
                bitmapBuffer.width,
                bitmapBuffer.height,
                rotationMatrix,
                false
            )
        }

    }
```

## 运行效果图

#### 扫描玫瑰

<img src="pic\rose.jpg" width="40%" />

#### 扫描郁金香

<img src="pic\dup.jpg" width="40%" />