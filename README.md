# lenguajedesignos
"Lenguaje de signos" is a project designed to recognize and translate the Spanish sign language into text, to facilitate communication.
## How does it work?
After testing several approaches, we realized that the best way to implement this project was using a CNN (VggNet) wich will be fed not by
images of the signs, but by a pixel-wise mask that will extract de hands from the rest of the image. This mask has been made by making the
person that is signing wear gloves with a specific color. Then this color will be used as reference to obtain the mask.
## What does this project contain?
Inside this project you will file several files, that we can group in the following categories:
* **splitters**: the purpose of those files is to prepare de data to train yourself the network. 
They get a video or a folder with videos, divide them by frames and save those frames to disk. 
The ones with the word "mask" in their names also used the mask technique describe previously.
* **color picker**: this file is the one in charge of allow us to pick the color of the image that we want to use to separate the gloved 
hands from the rest of the image.
* **classifiers**: those files are the ones that you have to use to get the project working. They will get the model that already has been
trained with our data set and they will start classifying. One of them only clasify individual images, and the other get the video stream
of your webcam and starts classifying that video.
* **train**: with this file you will be able to train your model on your own data set if you do not want to use the one we provide
* **model**: the model trained with our dataset
* **labels**: labels obtained from our dataset
* **dataser folder**: in this folder we provide you the already masked images with which we have trained our network
* **pyimagesearch folder**: inside of which we have the structure of our network

## How to use it
The are two possibilities regarding the use of this project: use the model provided or train the network yourself.

If you only want to use this project with the already trained model, all you have to do is run the following command 
on your python console (being inside the project):

```
run classify-video.py -m guantes2.model -l guantes2.pickle
```
This command will start your webcam and start classifying in real time. Here we can see how it will be displayed:

![Imagen 1](/datasetguantes2/adios/0-1.png)

On the other hand, maybe you want to train the network with yor own dataset, or even train it with our dataset, but by yourself.
In this case you have to run the following command:
```
run train.py --dataset dataset --model output.model --outputlabel lb.pickle
```

This will train the network with the dataset provided and write de output model and the output labels where indicated
After that you can run the classify-video.py file as explained before and see the results.

In every file you can find a first section of comments in which there is explained how to use them.

## Authors

* **Henar Domínguez Elvira** - *user:*  [HenarDoel](https://github.com/HenarDoel)
* **Mario Alfonso Arsuaga** - *user:*  [marloquemegusta](https://github.com/marloquemegusta)
