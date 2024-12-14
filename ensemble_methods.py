"""
Using the three individual models in Step 2, implement the following three different ensemble methods:
Step 2 models are Alexnet, ResNet18, and VGG16.

a) Max probability: The SoftMax output of each model is considered as a probability. Use the
maximum value among the three to indicate the correct output label.

b) Probability averaging: Average the SoftMax output of each model, and use the maximum
average value to indicate the correct output label.

c) Majority voting: Consider the top result from each model to be a vote, and take the label with
the majority number of votes (i.e. 2 or 3) as the correct output label.
"""

#assume arrays of confidence values for each model are given as input

import numpy as np
import torch

"""Max probability: The SoftMax output of each model is considered as a probability. Use the
maximum value among the three to indicate the correct output label."""
def max_probability(alexnet, resnet18, vgg16):

    #softmax the arrays
    alexnet = alexnet.softmax(dim=0)
    resnet18 = resnet18.softmax(dim=0)
    vgg16 = vgg16.softmax(dim=0)

    max_prob_array = []
    for i in range(len(alexnet)):
        max_prob_array.append(max(alexnet[i], resnet18[i], vgg16[i]))

    max_prob = max(max_prob_array) #Get the highest confidence value from the three models
    label = max_prob_array.index(max_prob) #Get the index of the highest confidence value

    return label


"""Probability averaging: Average the SoftMax output of each model, and use the maximum
average value to indicate the correct output label."""
def probability_averaging(alexnet, resnet18, vgg16):

    #softmax the arrays
    alexnet = alexnet.softmax(dim=0)
    resnet18 = resnet18.softmax(dim=0)
    vgg16 = vgg16.softmax(dim=0)

    avg_prob_array = []
    for i in range(len(alexnet)):
        avg_prob_array.append((alexnet[i] + resnet18[i] + vgg16[i])/3)

    max_prob = max(avg_prob_array) #Get the highest confidence value from the three models
    label = avg_prob_array.index(max_prob) #Get the index of the highest confidence value

    return label

"""Majority voting: Consider the top result from each model to be a vote, and take the label with
the majority number of votes (i.e. 2 or 3) as the correct output label."""
def majority_voting(alexnet, resnet18, vgg16):
    

    #Get the index of the highest confidence value for each model
    alexnet_max_index = torch.argmax(alexnet).item()
    resnet18_max_index = torch.argmax(resnet18).item()
    vgg16_max_index = torch.argmax(vgg16).item()

    #Create a list of the highest confidence values
    index_list = [alexnet_max_index, resnet18_max_index, vgg16_max_index]
    index_set = set(index_list)

    #Find the most common index
    majority_vote = max(index_set, key = index_list.count)

    #if all indices are different, call probability_averaging
    if len(index_set) == 3:
        return probability_averaging(alexnet, resnet18, vgg16)

    return majority_vote


#ensemble functions for batches

def max_probability_batch(alexnet, resnet18, vgg16):

    labels = []

    for output in range(len(alexnet)):

        label = max_probability(alexnet[output], resnet18[output], vgg16[output])
        labels.append(label)

    #convert to tensor
    labels_tensor = torch.tensor(labels, device= alexnet.device)
    return labels_tensor

def probability_averaging_batch(alexnet, resnet18, vgg16):

    labels = []

    for output in range(len(alexnet)):
        
        label = probability_averaging(alexnet[output], resnet18[output], vgg16[output])
        labels.append(label)

    #convert to tensor
    labels_tensor = torch.tensor(labels, device= alexnet.device)
    return labels_tensor


def majority_voting_batch(alexnet, resnet18, vgg16):

    labels = []

    for output in range(len(alexnet)):

        label = majority_voting(alexnet[output], resnet18[output], vgg16[output])
        labels.append(label)

    #convert to tensor
    labels_tensor = torch.tensor(labels, device= alexnet.device)
    return labels_tensor