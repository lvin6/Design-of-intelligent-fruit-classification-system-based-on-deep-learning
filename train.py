import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import json
from fruits_classification_model import FruitsClassificationModel


num_classes = 131  # Total number of categories

training_dir = '..' + os.sep + 'dataset/Training'
validation_dir = '..' + os.sep + 'dataset/Validation'
dist_dir = ".." + os.sep + "dist"

start_time = time.time()  # Start time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load dataset
train_data = datasets.ImageFolder(training_dir, transform=transform)
test_data = datasets.ImageFolder(validation_dir, transform=transform)

batch_size = 48

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
print()
print('Size of Training dataset: ', (len(train_loader.dataset)))
print('Size of Testing dataset: ', (len(test_loader.dataset)))
print()

json_classes_labels = json.dumps(train_data.class_to_idx)
print()
print("categories labels below:")
print(json_classes_labels)  # print classes with labels in json format
print()

#   save categories labels to json file
with open("../dist/labels.json", "w+") as f:
    f.write(json_classes_labels)


def show_image(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))


data_iter = iter(train_loader)
images, labels = next(data_iter)

show_image(torchvision.utils.make_grid(images[0:8]))

print()
print('sample dataset image batch')
print(', '.join('%5s' % train_data.classes[labels[j]] for j in range(8)))
print()
print('please close the popup window to continue')
plt.show()

num_epochs = 4
alpha = 0.0001

cnn = FruitsClassificationModel()

print()
print('CNN Arch: ', cnn)
print()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=alpha)

# use gpu for training if cuda is available, otherwise cpu is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print()
cnn.to(device)

loss_list = []

# training started
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))

print()
print('Run time: %.3f seconds' % (time.time() - start_time))
print()

torch.save(cnn.state_dict(), dist_dir + os.sep + 'fruits_classification_model.pth')

print()
print('Start the verification process..')
print()

cnn.cpu()

correct = 0.0
total = 0.0

accuracy_list = []

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

# test started
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(8):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    accuracy = 100 * correct // total
    accuracy_list.append(accuracy.data)

for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        train_data.classes[i], 100 * class_correct[i] // class_total[i]))

print()
print('Accuracy: %.3f %%' % (100 * correct / total))
print()

a = input('Press enter to exit')
if a:
    exit(0)
