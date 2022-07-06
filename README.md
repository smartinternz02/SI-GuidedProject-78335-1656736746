# SI-GuidedProject-78335-1656736746
ECG- Image based Heartbeat classification for Arrhythmia Detection Using IBM Watson Studio

ECG- Image Based Heartbeat Classification For Arrhythmia Detection Using IBM Watson Studio

Introduction:

1.1 Overview:
According to the World Health Organization (WHO), cardiovascular diseases (CVDs) are the number one cause of death today. Over 17.7 million people died from CVDs in the year 2017 all over the world which is about 31% of all deaths, and over 75% of these deaths occur in low and middle-income countries. Arrhythmia is a representative type of CVD that refers to any irregular change from the normal heart rhythms. There are several types of arrhythmia including atrial fibrillation, premature contraction, ventricular fibrillation, and tachycardia. Although a single arrhythmia heartbeat may not have a serious impact on life, continuous arrhythmia beats can result in fatal circumstances. In this project, we build an effective electrocardiogram (ECG) arrhythmia classification method using a convolution al neural network (CNN), in which we classify ECG into seven categories, one being normal and the other six being different types of arrhythmia using deep two-dimensional CNN with grayscale ECG images. We are creating a web application where the user selects the image which is to be classified. The image is fed into the model that is trained and the cited class will be displayed on the webpage.

1.2 Purpose:
In the past few decades, Deep Learning has proved to be a compelling tool because of its ability to handle large amounts of data. The interest to use hidden layers has surpassed traditional techniques, especially in pattern recognition. One of the most popular deep neural networks is Convolution al Neural Networks.

In deep learning, a convolution al neural network (CNN/ConvNet) is a class of deep neural networks, most commonly applied to analyze visual imagery. Now when we think of a neural network we think about matrix multiplications but that is not the case with ConvNet. It uses a special technique called Convolution. Now in mathematics convolution is a mathematical operation on two functions that produces a third function that expresses how the shape of one is modified by the other.


Literature Survey:
2.1  Existing Problem:
Cardiovascular diseases (CVDs) are the number one cause of death today. Over 17.7 million people died from CVDs in the year 2017 all over the world which is about 31% of all deaths, and over 75% of these deaths occur in low and middle-income countries. Arrhythmia is a representative type of CVD that refers to any irregular change from the normal heart rhythms. There are several types of arrhythmia including atrial fibrillation, premature contraction, ventricular fibrillation, and tachycardia.



2.1  Proposed Solution:
An "ambulatory electrocardiogram" or an ECG) about the size of a postcard or digital camera that the patient will be using for 1 to 2 days, or up to 2 weeks. The test measures the movement of electrical signals or waves through the heart. These signals tell the heart to contract (squeeze) and pump blood. The patient will have electrodes taped to your skin. It's painless, although some people have mild skin irritation from the tape used to attach the electrodes to the chest.They can do everything but shower or bathe while wearing the electrodes. After the test period, patient will go back to see your doctor. They will be downloading the information.











Theoretical Experience:
3.1  Block Diagram:

We will prepare the project by following the below steps:
We will be working with Sequential type of modeling
We will be working with Keras capabilities
We will be working with image processing techniques
We will  build a web application using the Flask framework.
Afterwards we will be training our dataset in the IBM cloud and building another model from IBM and we will also test it.

3.2  HARDWARE & SOFTWARE Desgining:

Hardware Components used:
Since we are using the IBM cloud as a platform to execute this project we don’t need any hardware components other than our system.
Software Components Used:
We will be using Anaconda Navigator which is installed in our system and Watson studio from the IBM cloud to complete the project.
Anaconda Navigator
Anaconda Navigator is a free and open-source distribution of the Python and R programming languages for data science and machine learning related applications. It can be installed on Windows, Linux, and macOS.Conda is an open-source, cross-platform,  package management system. Anaconda comes with so very nice tools like JupyterLab, Jupyter Notebook,
QtConsole, Spyder, Glueviz, Orange, Rstudio, Visual Studio Code. For this project, we will be using Jupiter notebook and spyder

WATSON STUDIO:
Watson Studio is one of the core services in Cloud Pak for Data as a Service.
Watson Studio provides you with the environment and tools to solve your business problems by collaboratively working with data. You can choose the tools you need to analyze and visualize data, to cleanse and shape data, or to build machine learning models.
This illustration shows how the architecture of Watson Studio is centered around the project. A project is a workspace where you organize your resources and work with data.
Watson Studio projects fully integrate with the catalogs and deployment spaces:
Deployment spaces are provided by the Watson Machine Learning service
You can easily move assets between projects and deployment spaces.

Experimental Investigations:

In this project, we have deployed our training model using CNN on IBM Watson studio and in our local machine. We are deploying 4 types of CNN layers in a sequential manner , starting from :
Convolutional layer 2D:A 2-D convolutional layer applies sliding convolutional filters to 2-D input. The layer convolves the input by moving the filters along the input vertically and horizontally and computing the dot product of the weights and the input, and then adding a bias term.
Pooling Layer :Pooling layers are used to reduce the dimensions of the feature maps. Thus, it reduces the number of parameters to learn and the amount of computation performed in the network. The pooling layer summarises the features present in a region of the feature map generated by a convolution layer.
Fully-Connected layer :After extracting features from multiple convolution layers and pooling layers, the fully-connected layer is used to expand the connection of all features. Finally, the SoftMax layer makes a logistic regression classification. Fully-connected layer transfers the weighted sum of the output of the previous layer to the activation function.
Dropout Layer :There is usually a dropout layer before the fully-connected layer. The dropout layer will temporarily disconnect some neurons from the network according to the certain probability during the training of the convolution neural network, which reduces the joint adaptability between neuron nodes, reduces overfitting, and enhances the generalization ability of the network.
Flow Chart & Results with Screenshots:
5.1  Flow Chart & Results by training model in local machine:
Dataset Collection: 
The dataset contains six classes:
Left Bundle Branch Block
Normal
Premature Atrial Contraction
Premature Ventricular Contractions
Right Bundle Branch Block
Ventricular Fibrillation
Image Preprocessing:
Image Pre-processing includes the following main tasks
Import ImageDataGenerator Library:
Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset.

The Keras deep learning neural network library provides the capability to fit models using image data augmentation via the ImageDataGenerator class.



Configure ImageDataGenerator Class:
There are five main types of data augmentation techniques for image data; specifically:

Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotates  via the rotation_range argument
Image brightness via the brightness_range argument.
Image zooms via the zoom_range argument.

An instance of the ImageDataGenerator class can be constructed for train and test. 



Applying ImageDataGenerator functionality to the trainset and test set:
We will apply ImageDataGenerator functionality to Trainset and Testset by using the following code

This function will return batches of images from the subdirectories Left Bundle Branch Block, Normal, Premature Atrial Contraction, Premature Ventricular Contractions, Right Bundle Branch Block and Ventricular Fibrillation, together with labels 0 to 5{'Left Bundle Branch Block': 0, 'Normal': 1, 'Premature Atrial Contraction': 2, 'Premature Ventricular Contractions': 3, 'Right Bundle Branch Block': 4, 'Ventricular Fibrillation': 5}



We can see that for training there are 15341 images belonging to 6 classes and for testing there are 6825 images belonging to 6 classes.

Model Building
We are ready with the augmented and pre-processed image data,we will begin our build our model by following the below steps:
Import the model building Libraries:

Initializing the model:
Keras has 2 ways to define a neural network: 
Sequential
Function API 
The Sequential class is used to define linear initializations of network layers which then, collectively, constitute a model. In our example below, we will use the Sequential constructor to create a model, which will then have layers added to it using the add () method.
Now, will initialize our model.
Adding CNN Layers:
We are adding a convolution layer with an activation function as “relu” and with a small filter size (3,3) and a number of filters as (32) followed by a max-pooling layer.

The Max pool layer is used to downsample the input.

The flatten layer flattens the input. 

Adding Hidden Layers:
Dense layer is deeply connected neural network layer. It is most common and frequently used layer.

Adding Output Layer:

Understanding the model is very important phase to properly use it for training and prediction purposes. Keras provides a simple method, summary to get the full information about the model and its layers.


Configure the Learning Process:
The compilation is the final step in creating a model. Once the compilation is done, we can move on to the training phase. The loss function is used to find error or deviation in the learning process. Keras requires loss function during the model compilation process. 
Optimization is an important process that optimizes the input weights by comparing the prediction and the loss function. Here we are using adam optimizer
Metrics is used to evaluate the performance of your model. It is similar to loss function, but not used in the training process.


Training the model:
We will train our model with our image dataset. fit_generator functions used to train a deep learning neural network.

Saving the model:
The model is saved with .h5 extension as follows 
An H5 file is a data file saved in the Hierarchical Data Format (HDF). It contains multidimensional arrays of scientific data.

Testing the model:
Load necessary libraries and load the saved model using load_model
Taking an image as input and checking the results 
Note: The target size should for the image that is should be the same as the target size that you have used for training.


The unknown image uploaded is:

Here the output for the uploaded result is normal.

Application Building:
In this section, we will be building a web application that is integrated into the model we built. A UI is provided for the uses where he has uploaded an image. The uploaded image is given to the saved model and prediction is showcased on the UI.
This section has the following tasks
Building HTML Pages:
We use HTML to create the front end part of the web page. 
Here, we created 4 html pages- home.html, predict_base.html, predict.html, information.html. 


home.html displays the home page.


information.html displays all important details to be known about ECG. 


predict-base.html and predict.html accept input from the user and predicts the values.





Building server-side script:
We will build the flask file ‘app.py’ which is a web framework written in python for server-side scripting. 
The app starts running when the “__name__” constructor is called in main.
render_template is used to return HTML file.
“GET” method is used to take input from the user.
“POST” method is used to display the output to the user. 


Running The App:


Navigate to the localhost (http://127.0.0.1:5000/)where you can view your web page.








5.2  Flow Chart & Results by training model in IBM WATSON STUDIO:
Creating IBM cloud account:
We have to create an IBM Cloud Account and should log in.
Creting Watson Studio Service & Machine Learning Service:

Create a Project & Deployment space in the watson studio:





d.   Upload The dataset and create a jupyter source file in the created project:




e.   Apply CNN algorithm and save the model and deploy it using API key generated:



f.   For downloading the model we have to run the last part of the above code in the local jupyter notebook:



g.    Now we will extract the .h5 model file and will do the app deployment using flask as done in the previous training:

Hence we trained the model using IBM Watson.


Advantages & Disadvantages:
6.1  Advantages:
The proposed model predicts Arrhythmia in images with a high accuracy rate of  nearly 96%
The early detection of Arrhythmia gives better understanding of disease causes, initiates therapeutic interventions and enables developing appropriate treatments.
6.2  Disadvantages:
Not useful for identifying the different stages of Arrhythmia disease.
Not useful in monitoring motor symptoms



Applications :
It is useful for identifying the arrhythmia disease at an early stage.
It is useful in detecting  cardiovascular disorders
.
Conclusion:
Cardiovascular disease is a major health problem in today's world. The early diagnosis of cardiac arrhythmia highly relies on the ECG. 
Unfortunately, the expert level of medical resources is rare, visually identify the ECG signal is challenging and time-consuming. 
The advantages of the proposed CNN network have been put to evidence. 
It is endowed with an ability to effectively process the non-filtered dataset with its potential anti-noise features. Besides that, ten-fold cross-validation is implemented in this work to further demonstrate the robustness of the network.

Future Scope:
For future work, it would be interesting to explore the use of optimization techniques to find a feasible design and solution. The limitation of our study is that we have yet to apply any optimization techniques to optimize the model parameters and we believe that with the implementation of the optimization, it will be able to further elevate the performance of the proposed solution to the next level.

References:
https://github.com/Anshuman151/ECG-Image-Based-Heartbeat-Classification-for-Arrhythmia-Detection-Using-IBM-Watson-Studio/blob/main/README.md
https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/
https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.convolution2dlayer.html;jsessionid=0a7e3bc26fabda07a5032030294b

[Youtube Link] for reference(https://www.youtube.com/watch?v=EQG7rN2R-tc)
    
                                                 THE END


                                                                                            Name: Mandla Sheshi Kiran Reddy
                                                                                                Email-id: sheshikiranmandla2589@gmail.com

