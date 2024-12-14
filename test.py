import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
# Import CIFAR100 dataset
from torchvision.datasets import CIFAR100

import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


weights_AlexNet_5 = 'weights_AlexNet_5.pth'
weights_VGG16_5 = 'weights_VGG16_5.pth'
weights_ResNet18_5 = 'weights_ResNet18_5.pth'

weights_AlexNet_fc = 'weights_AlexNet_fc.pth'
weights_VGG16_fc = 'weights_VGG16_fc.pth'
weights_ResNet18_fc = 'weights_ResNet18_fc.pth'


def main():
    # Default values
    chosen_model = "a"
    batch_size = 32
    num_epochs = "5"

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=str, help='model to use [AlexNet, VGG16, ResNet18]', default='AlexNet')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]', default=32)
    argParser.add_argument('-e', metavar='num epochs', type=str, help='number of epochs [5, fc]', default='5')

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.m != None:
        chosen_model = args.m
    if args.b != None:
        batch_size = args.b
    if args.e != None:
        num_epochs = args.e

    print("Running testing")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    model = None
    weights_file = 'weights.pth'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if chosen_model == 'AlexNet' or chosen_model == 'a':
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, 100)

        if num_epochs == "5":
            weights_file = 'weights_AlexNet_5.pth'
        elif num_epochs == "fc":
            weights_file = 'weights_AlexNet_fc.pth'

    elif chosen_model == 'VGG16' or chosen_model == 'v':
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 100)

        if num_epochs == "5":
            weights_file = 'weights_VGG16_5.pth'
        elif num_epochs == "fc":
            weights_file = 'weights_VGG16_fc.pth'

    elif chosen_model == 'ResNet18' or chosen_model == 'r':
        model = models.resnet18()
        model.fc = nn.Linear(512, 100)

        if num_epochs == "5":
            weights_file = 'weights_ResNet18_5.pth'
        elif num_epochs == "fc":
            weights_file = 'weights_ResNet18_fc.pth'
    else:
        print("Invalid model chosen")
        return

    model.to(device)

    #load save file
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_file))
    else:
        model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))

    model.eval()

    #Load datasets
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)

    #Create DataLoaders
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    num_batches = len(test_loader)
    num_samples = len(test_set)

    print("Testing on ", num_batches, " batches")

    #calculate cross entropy loss
    calculate_loss = nn.CrossEntropyLoss()

    loss = 0
    top1_error = 0
    top5_error = 0
    count = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):

            #get the inputs and labels
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)

            #calculate loss
            loss += calculate_loss(outputs, labels).item()

            #calculate top 1 error
            _, predicted = torch.max(outputs, 1) #get the index of the max log-probability (value can be ingored(_))
            top1_error += (predicted != labels).sum().item() #sum up the errors

            #calculate top 5 error
            _, top5 = torch.topk(outputs, 5, dim=1) #get the top 5 predictions (value can be ingored(_))
            for j in range(len(labels)): #for each label
                if labels[j] not in top5[j]: #if the label is not in the top 5 predictions
                    top5_error += 1 #increment the error

            count += 1

    #calculate average loss, top 1 and top 5 error rates
    loss /= num_batches
    top1_error /= num_samples
    top5_error /= num_samples


    print("Average loss: ", loss)
    print("Top 1 error rate: ", top1_error)
    print("Top 5 error rate: ", top5_error)

    return loss, top1_error, top5_error


if __name__ == '__main__':
    main()
