import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

# Import CIFAR100 dataset
from torchvision.datasets import CIFAR100

import argparse
import ensemble_methods

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Default values
    ensemble_method = "max"
    batch_size = 32
    num_epochs = "5"

    # read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=str, help='ensemble method to use[max, avg, vote]', default='max')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size [32]', default=32)
    argParser.add_argument('-e', metavar='num epochs', type=str, help='number of epochs [5, fc]', default='5')

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.m != None:
        ensemble_method = args.m
    if args.b != None:
        batch_size = args.b
    if args.e != None:
        num_epochs = args.e

    print("Running testing")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    alexnet_model = models.alexnet()
    alexnet_model.classifier[6] = nn.Linear(4096, 100)

    vgg16_model = models.vgg16()
    vgg16_model.classifier[6] = nn.Linear(4096, 100)

    resnet18_model = models.resnet18()
    resnet18_model.fc = nn.Linear(512, 100)


    if num_epochs == "5":
        alexnet_weights_file = 'weights_AlexNet_5.pth'
        vgg16_weights_file = 'weights_VGG16_5.pth'
        resnet18_weights_file = 'weights_ResNet18_5.pth'

    elif num_epochs == "fc":
        alexnet_weights_file = 'weights_AlexNet_fc.pth'
        vgg16_weights_file = 'weights_VGG16_fc.pth'
        resnet18_weights_file = 'weights_ResNet18_fc.pth'

    else:
        print("Invalid number of epochs")
        return

    alexnet_model.to(device)
    vgg16_model.to(device)
    resnet18_model.to(device)

    #load save file
    if torch.cuda.is_available():
        alexnet_model.load_state_dict(torch.load(alexnet_weights_file))
        vgg16_model.load_state_dict(torch.load(vgg16_weights_file))
        resnet18_model.load_state_dict(torch.load(resnet18_weights_file))
    else:
        alexnet_model.load_state_dict(torch.load(alexnet_weights_file, map_location=torch.device('cpu')))
        vgg16_model.load_state_dict(torch.load(vgg16_weights_file, map_location=torch.device('cpu')))
        resnet18_model.load_state_dict(torch.load(resnet18_weights_file, map_location=torch.device('cpu')))

    alexnet_model.eval()
    vgg16_model.eval()
    resnet18_model.eval()

    #Load datasets
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)

    #Create DataLoaders
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    num_batches = len(test_loader)
    num_samples = len(test_set)

    print("Testing on ", num_batches, " batches")

    count = 0
    top1_error = 0

    with torch.no_grad():
        for data in test_loader:
            count += 1

            #get the inputs and labels
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device) 
            
            #get the outputs
            alexnet_outputs = alexnet_model(inputs)
            vgg16_outputs = vgg16_model(inputs)
            resnet18_outputs = resnet18_model(inputs)

            #Apply ensemble method
            if ensemble_method == "max":
                outputs = ensemble_methods.max_probability_batch(alexnet_outputs, vgg16_outputs, resnet18_outputs)
            elif ensemble_method == "avg":
                outputs = ensemble_methods.probability_averaging_batch(alexnet_outputs, vgg16_outputs, resnet18_outputs)
            elif ensemble_method == "vote":
                outputs = ensemble_methods.majority_voting_batch(alexnet_outputs, vgg16_outputs, resnet18_outputs)
            else:
                print("Invalid ensemble method")
                return

            top1_error += (outputs != labels).sum().item() #Get number of incorrect predictions and add to total

            if count % (num_batches // 10) == 0 or count == 1:
                print("Batch ", count, " top 1 error: ", top1_error/(batch_size*count), " total incorrect: ", top1_error)

            

    #calculate top1 error rate
    top1_error = top1_error / num_samples

    print("Top 1 error rate: ", top1_error)

    return top1_error


if __name__ == '__main__':
    main()