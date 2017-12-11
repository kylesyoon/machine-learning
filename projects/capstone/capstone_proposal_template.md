# Machine Learning Engineer Nanodegree
## Capstone Proposal
Kyle Yoon  
December 10th, 2017

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

Deep Learning has become a significant field for Human-Computer Interaction (HCI). The interface for interacting with a computer has changed from just keyboards and mice to various touch gestures on a mobile device and motion wands for augmented reality. The applications are in various fields, including virtual environments, smart surveillance and Human-Computer Intelligent Interaction[1].

Based on the application hand gestures can be categorized into the following categories[1].
- Conversational gestures
- Controlling gestures
⋅⋅* Focus for current vision-based interface
⋅⋅* Display-control applications, navigation gestures like pointing
- Manipulative gestures
⋅⋅* Tele-operation and virtual assembly
- Communicative gestures
⋅⋅* Sign language

An example of vision-based HCI application systems are:
- An automatic system for analyzing and annotating video sequences of technical talks[2]. 
- FingerPaint application, which recognizes finger movements as input for augmented reality[3].

The motivation behind this proposal is to bring the interaction between Tony Stark and Jarvis, from the fictional Marvel movie series Ironman, to closer to reality. The beginning of the mentioned [clip](https://youtu.be/DZaAFADoF1M) from the movie _Ironman_ shows an example. 

### Problem Statement
_(approx. 1 paragraph)_

This project is an attempt to obtain high accuracy for the [VIVA Hand Detection Challenge](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/). More specifically, the challenge will be to train a successful CNN that also uses sliding search windows and heat mapping.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

The dataset for the VIVA Hand Detection Challenge has 5500 training images and 5500 testing images. The images are the entire front space of cars with a driver in a natural setting with varying illumination and hand movements. Bounding boxes of the hands are provided for each training image. These bounding boxes will be used to produce a hand dataset that has positive hand images and negative non-hand images.

The inputs include the model itself. Such inputs can be a LeNet or an ImageNet weighted Xception model with transfer learning. In other words, the layers of the CNN are a critical input. Compiling the model has various inputs as well such as loss function, optimizer and metrics. Fitting the model has epoch count and batch size inputs.

Finally, parameters regarding the sliding search window are inputs as well. Examples would be the window size, step size and threshold for heat mapping.

### Solution Statement
_(approx. 1 paragraph)_

A CNN model will be trained to recognize the human hand from the VIVA Hand Detection Challenge. Many popular models will be compared to produce the highest accuracy. An example is the LeNet-5. However, this project will focus on using transfer learning on established models, such as Xception or ResNet50, with ImageNet weights.

Additionally, a sliding search window method will be used in order to detect the hand. Heat mapping will be applied to narrow down and consolidate the detected hand images.

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

There are several benchmark models that are available for the VIVA Hand Detection Challenge, the top ranking one being the Multiple-scale Deep Convolution Neural Network for Fast Object Detection. Others include the popular YOLO model. 

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

The evaluation metric that will be used is accuracy. This accuracy will be based on the hand dataset created from the VIVA Hand Detection Dataset. In other words, it is the accuracy of the model predicting images that are either labeled as a positive hand image or a negative hand image.

However, another evaluation that is required is the accuracy of the bounding box created from the heat mapped search windows. Evaluating whether the drawn box accurately frames a hand in an image is subjective and difficult to quantify. This is because the frame can vary widely, while still be considered a successful bounding box for the hand. Therefore, this evaluation will be done manually by processing separate images that contain hands and viewing the result.

### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

The project workflow begins with additional research on the benchmark models. While region-based CNNs (R-CNN) are not in the scope of this project, the top ranked benchmark models are mainly R-CNNs.

Then dataset collection and preprocessing will take place. The VIVA Hand Detection Challenge provides images and bounding boxes. The CNN for this project is not region-based, therefore the images must be proprocessed to be cropped to the bounding box. When preprocessing each image, a negative non-hand image will also be cropped by randomly selecting a frame that is equivalent to the bounding box size, but outside. In order to achieve this, a generator will be created. The generator will preprocess the image, yielding postive and negative labeled images on the fly. 

The model selection will take an iterative approach. Established CNN architectures will trained and evalutated. Examples of these architectures are LeNet-5 and AlexNet. Another approach for selecting the models is transfer learning. Keras provides many models with ImageNet pre-trained weights. Bottleneck features will be extracted then used to train the models. Examples of these models are Xception, VGG16, VGG19, ResNet 50, InceptionV3, InceptionResNetV2 and MobileNet. Submodules for each model will be written so that training and evaluation for each model can be done repetitively. The architecture of the project package will look like:

```
|--- data
...
|--- output
|--- viva
|--- |--- __init__.py
|--- |--- cnn
|--- |--- |--- __init__.py
|--- |--- |--- networks
|--- |--- |--- |--- __init__.py
|--- |--- |--- |--- lenet.py
|--- |--- |--- |--- xception_transfer.py
...
|--- |--- bottleneck
|--- |--- |--- extract_bottlenect_features.py
...
|--- viva_pipeline.py

```
[5]

As shown in the simple mapping of the project architecture, a `viva_pipeline.py` file will be made. This pipeline file will be the code that glues the separate components together. Arguments will be added to the python file so that options can be easily changed when running from the command line. The output of the pipeline will the the accuracy as well as the saved weights for that particular run. 

Such process will be repeated. Models and parameters (e.g. batch size, epoch step size, number of epochs, image size) will be continuously tweaked until a satisfactory accuracy is achieved.

Then the project will move into the sliding search window phase. A window will be slided across a test image and return a list of subwindows. The model will predict a hand in these subwindows. Once marked, a window will provide an increment to the heat. The areas that pass the heat threshold will have a bounding box drawn. 

The search windows allow for a brute force method of achieving hand detection. Window size, step size and other paramters will be modified until several test images have bounding boxes around the hands.


### Resources

1. [Wu, Y., Huang, T.: Vision-Based Gesture Recognition: A Review, Beckman Institue University of Illinois at Urbana-Champaign](http://ai2-s2-pdfs.s3.amazonaws.com/59fe/0f477f81a8671956b8d1363bdc06ae8b08b3.pdf)
2. Ju, S., Black, M., Minneman, S., Kimber, D.: Analysis of Gesture and Action in Technical Talks for Video Indexing, IEEE Conf. on Computer Vision and Pattern Recognition, CVPR97. (1997)
3. Crowley,J., Berard,F., Coutaz,J.: Finger Tracking as An Input Device for Augmented Reality, Int.Workshop on Automatic Face and Gesture Recognition, Zurich, pp.195-200. (1995)
4. https://www.scribd.com/doc/98199282/Finger-Tracking-in-Real-Time-Human-Computer-Interaction
5. [Adrian Rosebrock: LeNet - Convolutional Neural Network in Python](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)
