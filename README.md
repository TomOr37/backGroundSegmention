# Research Project
### Subject: Detecting background in static image.
### By: Tom Or


### Task Description: 
Detecting the background in a given image is a problem that arises in many applications, such as object recognition, image editing, and video analysis. The goal is to separate the foreground objects from the background in an image, allowing the foreground objects to be analyzed and manipulated independently.
It is a classification problem, as we need to classify which pixel is background/foreground.

However, detecting the background in an image can be challenging due to several factors, such as:

1. Complex background patterns: The background can have complex patterns that are difficult to distinguish from the foreground objects, making it challenging to separate the two.

2. Overlapping objects: In some images, foreground objects can overlap with the background, making it difficult to determine the exact boundary between the foreground and background regions.

3. Static: In static images (i.e images and not videos), there are no moving objects - separation between moving objects, and the background itself.

4. Occlusions: Foreground objects can occlude parts of the background, making it difficult to determine the background regions in these areas.

These challenges make detecting the background in a given image a difficult and time-consuming task, and it can be especially challenging to do this automatically and accurately. This is where machine learning can be a useful tool, as it can be trained to automatically identify the patterns and features that distinguish the foreground and background regions in an image.


### Different approaches for the problem

#### 1. Train a classifier from scratch:
Creating an image classifier from scratch involves several steps, including gathering and preprocessing the data, building and training the model,
and evaluating its performance.High-level overview of the process:

1. Data collection: Collect a large and diverse dataset of images that are labeled with their corresponding classes. The dataset should be representative of the real-world problem you are trying to solve.

2. Data preprocessing: Preprocess the data to prepare it for use in the model. This may include resizing the images, normalizing the pixel values, and converting the images to a format that can be used as input to the model.

3. Model building: Choose a suitable model architecture, that is well-suited for image classification tasks. Then, initialize the model and define its architecture, including the number of layers, the types of layers, and the parameters of each layer.

4. Model training: Train the model on the preprocessed data using a suitable optimization algorithm, such as stochastic gradient descent (SGD) . The goal is to find the best set of parameters for the model that minimize the error between the model's predictions and the true labels.

5. Model evaluation: Evaluate the performance of the model on a separate dataset that was not used in the training process. This will give you an idea of how well the model generalizes to new, unseen data. Metrics such as accuracy, precision, recall, and F1 score can be used to evaluate the model's performance.

This approach is complected, as every step can be costly and not trivial to execute. 

#### 2. Fine-tune a pre-trained model.
It is possible to take a pre-trained model for an image segmentation, and add a classifier head (add a linear layer as the last layer).
Then train the pre-trained model on labeled dataset as described in steps 1,2 above. This approach was not taken, as I couldn't find a suitable data-set.

#### 3. Adding "rules" to a pre-trained model.
This approach resembles approach `2` , but slightly different. Instead of adding another head, adding rules to the output of the existing model: \
The model output is an index of pre-defined class for every pixel. We can leverage that to define 2 types of rules that will allow us to extract the background. \
*First Rule*: Define in advance which classes are considered as "background", and take only them as the output of the background. (same with the foreground) \
*Second Rule*: A common sense is that the most repeated class is the "main object". So we can take everything else as the background, and the "main object" as the foreground.\

The First Rule will work well if the model was trained on many classes and capable of identifying them. The Second rule will work well in some cases, but doesn't require
Many assumption on the model itself, only the ability to separate between different objects (not needing the classes themselves) \
*This is the approach that was taken in this project*. There will be represented results of the 2 rules. \

#### 4. Using Multi-modal models

On the recent advancement in Natural Language Understanding (Such as ChatGPT) , many models combining text and images were published. With smart prompting engineering
and with the right model, one can extract the background and the foreground. This approach wasn't researched in this project, but I think that is an interesting lead and could make some good results, that's why
I decided to mention this method.


#### 5. Using depth estimation model

With a powerful depth model, one can try to extract pattern/rule to separate the background and the foreground. \
I didn't continue with this approach as I couldn't find enough resources on depth estimation from static images. 


### Approach taken

As mentioned, we will observe the rule based approach. Actually, it is 2 different approaches, corresponding to each rule. 

#### Model Chosen

The model chosen is `Segformer`, paper: https://arxiv.org/pdf/2105.15203.pdf. \

SegFormer consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on semantic segmentation benchmarks such as ADE20K and Cityscapes (segmentiation datasets)
.The hierarchical Transformer is first pre-trained on ImageNet-1k, after which a decode head is added and fine-tuned altogether on a downstream dataset. The specific model we will examine is `segformer-b5-finetuned-ade-640-640` 
which is a Segformer fine-tuned on the *ade20k* dataset. \
The *ade20k* dataset contains 20,210 images, each annotated with 150 semantic labels, including objects and stuff classes such as sky, building, wall, and floor. The images were collected from a diverse range of scenes, including both indoor and outdoor environments, and the annotations provide a detailed and rich representation
of the scene structure and composition. List of the labels as csv file can be found [here](https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv)

We will be using the **[model]**(https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) from the `huggingface` hub.

#### First rule - Using the predefined classes

As mentioned, the model was fine-tuned on the ade20k dataset, meaning the segmentation classes will be accordingly to the dataset, with the 150 labels.
Some labels can be regarded as "background" labels - such as grass,sky and more. Fortunately , in the published dataset, it is noted what labels can be regarded as such (the *Stuff* column in the csv above). \
To extract the background using this rule , we take the model output , and build a tensor mask with the same shape as the image. For every pixel index corresponding to "background label", we define in the mask at the same position the value `[0,0,0]`,otherwise we define the value `[1,1,1]`. 
Then, by multiplying element wise, the original image and tensor mask, we get the original picture with the background blackened (The original image is assumed to be in RGB format, therefore the three values of the mask). /
This approach works decent, but it can fail on some occasions:
1. If the model didn't detect correctly the classes , some parts of the foreground can be blackened and vise-versa.
2. If the image contains several objects that are "not background labeled", but does appear in the background, they will not be removed.


#### Second rule - Taking the most common object

A reasonable assumption about the foreground of a given picture is that the foreground takes most of the picture (of course it is not always true, but it is common, especially in selfies and when the foreground is closed to the lens)
To extract the background using this rule , we take the model output ,and calculate the most common label. Then we build a tensor mask similar to the other approach:  For every pixel index that is not labeled as the most repeated, we define in the mask at the same position the value `[0,0,0]`,otherwise we define the value `[1,1,1]`. 
Then we multiply as before. /
This approach also works decent, but it can fail on some occasions:
1. If the foreground contains couple of labels, it can blacken some parts of the foreground. An example to that is a person holding a dog - if the person takes most of the picture, the dog will be blackened. This problem can be solved using geometry (similar to convex-hall):
   getting the shape made by the most common label, and not blackening the pixels that are within the shape.
2. If the images contains several objects with the same class in the background and in the foreground. For example a person taking a selfie with another person in the background. The other person will not be blackened. This problem can be solved using a different pre-trained model 
that has the capability to identify different class instances in the same image (segment differently person1 and person2).
 


### Results

For the results, I will show some examples that the background extraction was successfully, and some examples that demonstrate the problems with the approach mentioned above.
In general , Some images will be "easier" to separate, and some will be harder.\
We will consider the following examples:

![](data/selfie.jpg) 
###### Selfie image from stock. "Perfect settings"
*** 


![](data/WIN_20221020_16_58_05_Pro.jpg)
###### Picture of me I have taken with my computer camera. Not perfect setting.
***
![](data/WIN_20221020_17_47_45_Pro.jpg)
###### Picture of me I have taken with my computer camera. Not in a perfect setting. Unique object that was not common in the data set ( 0.0026 of the ada20k dataset is lamps images).
***


#### First approach:

![](results/first_approach/selfie.png)

#### Second approach:

![](results/second_approach/selfie.png)


Both results are decent, as expected. The image settings are good: not many objects, the separation between the foreground and the background is clear, and the model is capable of classifying person (0.0160 of the model training data).

#### First approach:

![](results/first_approach/img.png)

#### Second approach:

![](results/second_approach/img.png)

Here we can see a major difference between the approaches. We see the weakness of the first approach, as "window", "chair" and "lamp" are not always part of the foreground.



#### First approach:
![](results/first_approach/failed_first_2.png)


#### Second approach:

![](results/second_approach/img_1.png)

Here both results are questionable. We see that the model is not perfect, and has some flaws, it didn't segment the lamp correctly, and it damaged the performance. We see from the second approach, that even if the model managed to correctly segment the lamp, we will still get the wrong
results as the chairs are taken larger part of the image.


### Summery

This project presented two rule based approaches to separate the background from the foreground in a given images. Both approaches uses a pre-trained segmentation model and apply rules to the output. The first approach was to define some classes as "background", and mask the background accordingly. 
The second approach was to take the most common label as the foreground, and consider the rest and the background. \
We mentioned the problems in those approaches, mainly as the approaches do not generalize well to "harder" scenes and images and also rely on the performance of the model. \
However with some inductive bias (image with a clear separation between the background and the foreground, not so many objects - which is common), the result are decent and even good, as we can see from the first example. 


### How to use 

After installing the dependencies from the requirements.txt , run from main.py with `Image path/Image url` as the first argument, and `False/True` as the second argument weather to the first approach, or the second approach.

All the can be found on github, [here](https://github.com/TomOr37/backGroundSegmention).

