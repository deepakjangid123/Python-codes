Implementation for Neural Network Dropout, in which it drops out some nodes of the network. \
Dropping out can be seen as temporarily deactivating or ignoring neurons of the network. This technique is applied in the training phase to reduce overfitting effects. \
Overfitting is an error which occurs when a network is too closely fit to a limited set of input samples.

# Basic Idea
The basic idea behind dropout neural networks is to dropout nodes so that the network can concentrate on other features. \
Think about it like this. You watch lots of films from your favourite actor. At some point you listen to the radio and hear somebody in an interview. \
You don't recognize your favourite actor, because you have seen only movies and your are a visual type. Now, imagine that you can only listen to the audio tracks of the films. \
In this case you will have to learn to differentiate the voices of the actresses and actors. So by dropping out the visual part you are forced to focus on the sound features!

# DataSet
In this particular also, I used the same MNIST DataSet as mentioned [here](https://github.com/deepakjangid123/Python-codes/tree/master/neural_networks).