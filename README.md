# yolo-object-detection-pipeline
This project implements an end-to-end YOLOv8 object detection pipeline. It demonstrates the complete workflow of loading a pretrained model, training it on a sample dataset, saving optimized weights, and performing inference on both images and real-time webcam input.

The pipeline uses transfer learning by initializing the model with pretrained weights and refining them through training on a small dataset. During training, the model learns to predict bounding boxes, objectness scores, and class probabilities by minimizing a composite loss function. The best-performing model weights are saved and later used for inference.

After training, the model is applied to detect objects in images, producing bounding boxes with confidence scores. The same inference process is extended to real-time webcam input, enabling continuous object detection.

The project also generates a structured output directory containing training logs, performance metrics, model checkpoints, and prediction results, providing a complete view of the model lifecycle from training to deployment.
