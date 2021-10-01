import argparse
import os
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


      
        
def data_transform(data_dir):
    
    train_dir, test_dir, valid_dir = data_dir 
    data_transforms = transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])

    resized_transforms = transforms.Compose([
                                  transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])    
    train_datasets = datasets.ImageFolder(train_dir, transform=resized_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=resized_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=resized_transforms)

    
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    loaders = {'train':trainloaders,'valid':validloaders,'test':testloaders,'labels':cat_to_name}
    return loaders


def check_parameters():
    print("validating parameters")
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--you dont have gpu enabled")
    if(not os.path.isdir(args.data_directory)):
        raise Exception('there is no such directory')
    data_dir = os.listdir(args.data_directory)
    if (not set(data_dir).issubset({'test','train','valid'})):
        raise Exception('data directory is missing')
    if args.arch not in ('vgg','densenet',None):
        raise Exception('You have to choose one of vggnet or densenet as pretrained model')

def fetch_data():
   
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    data_dir = [train_dir,test_dir,valid_dir]
    return data_transform(data_dir)

def load_model(data):
    print("create model object")
    if (args.arch is None):
        arch_type = 'vgg'
    else:
        arch_type = args.arch
    if (arch_type == 'vgg'):
        model = models.vgg19(pretrained=True)
        input_node=25088
    elif (arch_type == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_node=1024
    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_node, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

  

def train_model(model,data):
    print("training model")
    
    print_every=40
    
    if (args.learning_rate is None):
        learn_rate = 0.001
    else:
        learn_rate = args.learning_rate
    if (args.epochs is None):
        epochs = 5
    else:
        epochs = args.epochs
    if (args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'
    
    learn_rate = float(learn_rate)
    epochs = int(epochs)
    
    trainloader=data['train']
    validloader=data['valid']
    testloader=data['test']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    
    steps = 0
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()     
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_accuracy = test_accuracy(model,validloader,device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))            
                running_loss = 0
    print("Training completed")
    test_result = test_accuracy(model,testloader,device)
    print('final accuracy on test set: {}'.format(test_result))
    return model

def test_accuracy(model,loader,device='cpu'):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_model(model):
    print("saving model")
    if (args.save_dir is None):
        save_dir = 'model_checkpoint.pth'
    else:
        save_dir = args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}
    torch.save(checkpoint, save_dir)
    return 0

def create_model():
    check_parameters()
    data = fetch_data()
    model = load_model(data)
    model = train_model(model,data)
    save_model(model)
    return None

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='models to use OPTIONS[vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--gpu',action='store_true', help='gpu')
    args = parser.parse_args()
    return args

def main():
    print("creating a deep learning model")
    global args
    args = parse()
    create_model()
    print("model has completed training")
    return None

main()