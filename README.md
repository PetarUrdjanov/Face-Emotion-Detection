# Face Detection and Emotion Recognition [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
- Tech/Fameworks used: OpenCV, Keras/TensorFlow, OpenCV and Deep Learning (look at the requirements.txt file)

## Introduction
Human emotion detection is implemented in many areas requiring additional security or information about the person. Another important domain where we see the importance of emotion detection is for business promotions. Most of the businesses thrive on customer responses to all their products and offers. Also,the use of machines in society has increased widely in the last decades such as robots and in this context, deep learning has the potential to become a key factor to build better interaction between humans and machines, while providing machines with some kind of self-awareness about its human peers, and how to improve its communication with natural intelligence. 

> **Motivation and goals**

This project is a part of Data Science Academy by [Brainster](https://brainster.co/) with the primary goal, students to practically demonstrate the acquired knowledge as a final assignment. Technically, the project’s goal consists on training a deep neural network with labeled images of facial emotions. Finally, this is a multidisciplinary project involving affective computing, machine learning and computer vision. Learning how these different fields are related, and to understand how they can provide solutions to complex problems is another project’s goal. 

## 💡: Materials and Methods
The task in this project is to create a robust Image classifier, that given an image will find all of the faces within the image and then recognize the emotion of the person. The **:seven:emotions** besides neutral class that classifier will need to choose from are:

<img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-contempt.jpg?width=600&name=emotion-contempt.jpg" width=200 hight=200> <img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-happiness.jpeg?width=595&name=emotion-happiness.jpeg" width=200 hight=200>
<img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-sadness.jpg?width=595&name=emotion-sadness.jpg" width=200 hight=200> <img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-fear.jpg?width=519&name=emotion-fear.jpg" width=200 hight=200>
<img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-disgust.jpg?width=519&name=emotion-disgust.jpg" width=200 hight=200> <img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-anger.jpg?width=519&name=emotion-anger.jpg" width=200 hight=200> <img src="https://blog.trginternational.com/hs-fs/hubfs/images/emotion-surprise.jpg?width=600&name=emotion-surprise.jpg" width=200 hight=200>

We can generalize the emotion detection steps as follows:
- *Dataset preprocessing*
- *Face detection*
- *Feature extraction*
- *Classification based on the features*

This project has been divided into three phases: 

> **:one: phase** focused on dataset creation.More details about can be found in Chapter **Data**.

> **:two: phase** focused on training the model and consisted on the use of a facial emotion labeled data set to train a deep learning network. A detailed explanation is presented in Chapter **Models**

> **:three: phase** is focused on testing the model performance, explanation is presented in Chapter **Results**

### 📂: Data
- General Project Research

As it was stated previously, the following project belongs to the supervised learning category. The need for a data set containing images of facial emotions and their corresponding label is crucial. One of the first challenges while doing this project was collecting the data. We decided to collect images using ready-made data and datasets that were created for facial recognition purposes. The reason for this is the topic-specific, recognizing emotions. For this purpose, a couple of data sets were chosen to perform the experiment:

**DA NABROIME KOI DATASET GI KORISTEVME**
_______________________________
- Dataset Collection

This dataset consists of **35887 images** belonging to seven classes:

| Dataset         | Total      |
| -------------   | -----------| 
| **0 - anger**   | 4953       | 
| **1 - disgust** |  547       | 
| **2 - fear**    | 5121       | 
| **3 - happines**| 8989       | 
| **4 - sadness** | 6077       |
| **5 - surprise**| 4002       | 
| **6 - neutral** | 6198       | 
| **Total**       |35887       |

 ![EmotionLabel](EmotionLabel.jpg)

### Preview of dataset (first 7 images from the 7 target categories)
![anger](Anger.jpg)
![disgust](Disgust.jpg)
![fear](Fear.jpg)
![happiness](Happiness.jpg)
![sadness](Sadness.jpg)
![surprise](Surprise.jpg)
![neutral](Neutral.jpg)
____________________________________
- Dataset Preparation

We split the dataset on **train 80% /test 10% /validation 10%** data with python code
```
df_train=df.loc[df['Usage']=='Training']
df_test=df.loc[df['Usage']=='PublicTest']
df_validation=df.loc[df['Usage']=='PrivateTest']

```
and we get images by classes ratio:
| Dataset         | Training   | Test    | Validation | Total  |
| -------------   | -----------| ------- | ---------- | -----  |
| **0 - anger**   |            |         |            | 4953   |
| **1 - disgust** |            |         |            |  547   |
| **2 - fear**    |            |         |            | 5121   |
| **3 - happines**|            |         |            | 8989   |
| **4 - sadness** |            |         |            | 6077   |
| **5 - surprise**|            |         |            | 4002   |
| **6 - neutral** |            |         |            | 6198   |
| **Total**       |            |         |            |35887   |


### 💻: Models

- Research of tools and libraries

In this part of the project, we research the libraries and neural networks which correspond the best with our needs to finish the tasks. Nowadays, various packages are available to perform machine learning, deep learning, and computer vision problems. 

**OpenCV** is an open-source a video and image processing library and it is used for image and video analysis, like facial detection, license plate reading, photo editing, advanced robotic vision, and many more. It is supported by different programming languages such as R, Python, etc. It runs on most platforms such as Windows, Linux, and macOS. It is a complete package which can be used with other libraries to form a pipeline for any image extraction or detection framework.

**Python** is a powerful scripting language and is very useful for solving statistical problems involving machine learning algorithms. It has various utility functions which help in pre-processing. Processing is fast and it is supported on almost all platforms.It provides the pandas and numpy framework which helps in manipulation of data as per our needs.

**Scikit-learn** is the machine learning library in python. It comprises of matplotlib, numpy and a wide array of machine learning algorithms.The algorithm it provides can be used for classification and regression problems and their sub-types.

**Jupyter Notebook** is the IDE to combine python with all the libraries we will be using in our implementation. It is interactive, although some complex computations require time to complete. Plots and images are displayed instantly. 

**NESTO KRATKO ZA TOA KOJ PRE-TRAIN MODEL SME GO ZEMALE PREDVID I ZASTO**

**NESTO KRATKO ZA TOA STO KORISTIME ZA VIDEO**

**OBJASNUVAME AKO SAKAME I SO KOD ZA CEKORITE PODOLU**
- Face detection 
- Feature extraction and classification
- Training, testing, validating (comparing faces)
- Compose neural network arhitecture


### 🔑: Results

## 👏: Authors

💪:Team members:
* [Petar Urdjanov](https://github.com/PetarUrdjanov)
* [Savica Nedelkovska](https://github.com/Savica23)
* [Marija Ilievska](https://github.com/MarijaIlievska)
* [Martin Krsteski](https://github.com/MartinKrsteski)

👌Team supervisor:
* [Viktor Domazetovski](https://github.com/ViktorDo1)
