# ONNX YOLOv8 Instance Segmentation


# Important
- The input images are directly resized to match the input size of the model. I skipped adding the pad to the input image (image letterbox), it might affect the accuracy of the model if the input image has a different aspect ratio compared to the input size of the model. Always try to get an input size with a ratio close to the input images you will use.

# Requirements

 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

# Installation
```
git clone https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation.git
cd ONNX-YOLOv8-Instance-Segmentation
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
You can convert the Pytorch model to ONNX using the following Google Colab notebook:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oDEKz8FUCXtW-REhWy5N__PgTPjt3jm9?usp=sharing)
- The License of the models is GPL-3.0 license: [License](https://github.com/ultralytics/ultralytics/blob/master/LICENSE)

# Original YOLOv8 model
The original YOLOv8 Instance Segmentation model can be found in this repository: [YOLOv8 Instance Segmentation](https://github.com/ultralytics/ultralytics)

# Examples

 * **Image inference**:
 ```
 python image_instance_segmentation.py
 ```

 * **Webcam inference**:
 ```
 python webcam_instance_segmentation.py
 ```

 * **Video inference**: https://youtu.be/8j-FjTsLctA
 ```
 python video_instance_segmentation.py
 ```

