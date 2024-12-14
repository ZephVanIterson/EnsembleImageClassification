import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime
import argparse

from torchvision import models

# Import CIFAR100 dataset
from torchvision.datasets import CIFAR100

# Variables
batch_size = 32
learning_rate = 1e-4
n_epochs = 20
model_name = None
weights_model_name = None

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, validation_loss_fn=None, validation_loader=None, save_file=None, plot_file=None):
    print('Training ...')

    losses_train = []
    losses_val = []

    # Training Loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        print('Epoch {}/{}'.format(epoch, n_epochs))
        epoch_loss = 0.0

        # Training set
        for imgs, labels in train_loader:
            optimizer.zero_grad()

            if device == 'cuda':
                imgs = imgs.cuda()
                labels = labels.cuda()

            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            # Add to epoch loss total
            epoch_loss += loss.item()

        # Add epoch loss to list for plotting
        losses_train += [epoch_loss / len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, epoch_loss / len(train_loader)))

        if validation_loss_fn is not None:
            # Validation set
            model.eval()
            val_loss = 0.0

            for imgs, labels in validation_loader:
                if device == 'cuda':
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                loss = validation_loss_fn(outputs, labels)
                val_loss += loss.item()

            losses_val += [val_loss / len(validation_loader)]

            print('Epoch {}, Validation loss {}'.format(
                epoch, val_loss / len(validation_loader)))

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        if save_file != None:
            if epoch == 5:
                torch.save(model.state_dict(), "weights_" + weights_model_name + "_" + str(epoch) + ".pth")
            elif epoch == n_epochs:
                torch.save(model.state_dict(), "weights_" + weights_model_name + "_fc.pth")
            else:
                torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()

            plt.plot(losses_train, label='train')
            if validation_loss_fn is not None:
                plt.plot(losses_val, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', weights_model_name + "_" + plot_file)
            plt.savefig(weights_model_name + "_" + plot_file)

def main():
    global model_name, n_epochs, weights_model_name

    # Get arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=str, help='model to use [AlexNet, VGG16, ResNet18]', default='AlexNet')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size', default=32)
    argParser.add_argument('-e', metavar='num epochs', type=int, help='number of epochs', default=30)

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.m != None:
        model_name = args.m
    if args.b != None:
        batch_size = args.b
    if args.e != None:
        n_epochs = args.e

    # Configure device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('USING DEVICE ', device)

    # Create variables
    if model_name == "alexnet" or model_name == "AlexNet" or model_name == "a":
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, 100)
        weights_model_name = "AlexNet"
    elif model_name == "vgg16" or model_name == "VGG16" or model_name == "v":
        model = models.vgg16()
        model.classifier[6] = nn.Linear(4096, 100)
        weights_model_name = "VGG16"
    elif model_name == "resnet18" or model_name == "ResNet18" or model_name == "r":
        model = models.resnet18()
        model.fc = nn.Linear(512, 100)
        weights_model_name = "ResNet18"
    else:
        print("No valid model selected")
        exit()

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    validation_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Create transform for dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get datasets
    train_set = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Training on ", len(train_loader), " batches for ", n_epochs, " epochs")

    # Train model
    train(n_epochs,
          optimizer,
          model,
          loss_fn,
          train_loader,
          scheduler,
          device,
          validation_loss_fn=validation_loss_fn,
          validation_loader=test_loader,
          save_file='weights.pth',
          plot_file='plot.png')

###################################################################

if __name__ == '__main__':
    main()
