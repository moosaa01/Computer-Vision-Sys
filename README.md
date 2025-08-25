# Computer-Vision-Sys
Machine Learning vision system to detect different IoT sensors
Introduction
The goal of this assignment was to develop an image classification system using machine
learning based upon a given topic. The topic that was chosen for this project was for the image
classification system to recognise different IoT sensors when being shown through a
webcam.The model should be able to recognise the chosen IoT sensors in environments with
differing backgrounds and lighting conditions. To complete this assignment a machine learning
model was developed utilizing a convolutional neural network (CNN). A CNN was chosen due to
its proficiency in image recognition and object detection. The following is an explanation of the
development of this model.
Data Collection and Preprocessing
The first step in the process was to obtain a dataset and then preprocess the data. The dataset
was structured into training, validation, and testing subsets. This allowed for effective model
training and evaluation. Images of three classes were used. The first being the DHT 11
temperature and humidity sensor, second being the MQ 135 Gas Sensor and the third being the
PIR Digital Motion Sensor. Pictures were taken in different backgrounds and at different angles
to ensure variability. This helped the model to recognise the sensors in different environments
and under different lighting conditions. This was done to ensure that the model could detect the
sensors through the webcam and in real- world conditions. According to (Ng and
Katanforoosh,2024) It is good practice to split data into 80% training, 10% validating and 10%
testing. In this project a similar ratio was used however a slightly larger validating amount was
used to try and combat overfitting. Resizing was done within the code to ensure all images were
standardized as the model requires uniform measurements for it to effectively learn from the
images.
Model Development
The initial question that needed to be addressed was which neural network architecture would
be most suitable to complete the project objectives. A convolutional neural network (CNN) was
chosen due to its proven effectiveness in image classification involving different backgrounds
and angles. The framework that was going to be used was the next decision that was made. For
this project PyTorch was chosen. After some research it was determined that PyTorch would be
the easiest framework to learn for a first time machine learning project. A pretrained model
‘ResNet’ was chosen to promote transfer learning. This coupled with the dataset of IoT images
that were manually provided made it easier for the machine to improve its accuracy and
decrease its loss in the training, validating and testing stages. Some data augmentation
techniques were attempted, but overall they seemed to mainly increase loss and did not improve
accuracy therefore they were removed. In hindsight the reason for this outcome could be that
the data augmentation values that were chosen were too drastic which may have made it too
challenging for the model to learn from. Finally using the OpenCv libraries such as Cv2 for
webcam functionality that labels the shown object as well as showing the confidence level of the
image detection system.
Evaluation
The model was evaluated based on its performance in the training,validating and testing
datasets. A relatively larger training set was used in the validation phase compared to the
testing phase to ensure that there would be improvement in the validation process as during
earlier attempts the model struggled to improve its accuracy when validating.
Results for the training and validation loss and accuracy for a length of 20 epochs with an optimizer of 0.0001:
Test Loss:0.0910
Test Accuracy:96.67%
