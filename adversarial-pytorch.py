import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision import transforms
import scipy
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy

# Verify if CUDA is there
is_cuda = torch.cuda.is_available()

def display_tensor(tensor_img, denormalize_before_display =False):

    fig, (ax1) = plt.subplots(1, 1, figsize = (8,8))
    fig.sca(ax1)
    if denormalize_before_display:
        # Denormalize
        newImage = torch.FloatTensor(3,299,299)
        meanTensor = torch.FloatTensor([0.485, 0.456, 0.406])
        stddevTensor = torch.FloatTensor([0.229, 0.224, 0.225])
        newImage[0] = torch.add(torch.mul(tensor_img[0], stddevTensor[0]), meanTensor[0])
        newImage[1] = torch.add(torch.mul(tensor_img[1], stddevTensor[1]), meanTensor[1])
        newImage[2] = torch.add(torch.mul(tensor_img[2], stddevTensor[2]), meanTensor[2])
        ax1.imshow(transforms.ToPILImage()(newImage))
    else:
        ax1.imshow(transforms.ToPILImage()(tensor_img))
    plt.show()

def classify (img, correct_class = None, target_class = None, show_image = False, normalize = False):

    # copy image to new object
    tensor_img = copy.deepcopy(img)

    # Normalize the image if required
    if normalize:
        # transpose vector data - this models expects color channels first...
        tensor_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_img)

    # add extra dimension (the model expects that with a single image...)
    tensor_img.unsqueeze_(0)
    # move tensor to GPU
    if is_cuda:
        tensor_img = tensor_img.cuda()
    # load tensor onto variable
    tensor_img = Variable(tensor_img)

    # run inference... = set mode to eval
    inception_net.eval()
    # Perform the evaluation - and keesp category only and sofmax it
    outputs = inception_net(tensor_img)[0]
    # Softmax the output
    outputs = softmax(outputs, dim=0)

    # Retrieve the sorted output with categories
    sorted_outputs, indices = torch.sort(outputs, descending=True) 
    # Convert outputs to numpy arrays
    indices = indices.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    topk = list(indices[:10])
    topprobs = outputs[topk]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,8))
    barlist = ax2.bar(range(10), topprobs)

    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')
    
    fig.sca(ax1)
    ax1.imshow(transforms.ToPILImage()(img))
    plt.sca(ax2)
    plt.ylim([0, 1.1])

    imagenet_json = "imagenet.json"
    with open(imagenet_json) as f:
        imagenet_labels = json.load(f)

    plt.xticks(range(10), [imagenet_labels[i][:15] for i in topk], rotation = 'vertical')
    fig.subplots_adjust(bottom = 0.2)
    if show_image:
        plt.show()
    
    return tensor_img

#Initialize random number generators
np.random.seed(1234)
torch.manual_seed(1234)

img_class = 281 # chainlink fence
target_img_class = 235 # guacamole
learning_rate_gd = 0.1
momentum = 0.9
epsilon = 0.002/255.0
loss_threshold = 1e-6
steps = 100
source_img_filename = 'cat.jpg'


# Load pre-trained model
inception_net = models.inception_v3(pretrained=True)
if is_cuda:
    inception_net.cuda()    # move model to GPU
    print("Model loaded on GPU : " + str(torch.cuda.get_device_name(0)))
else:
    print("Model loaded on CPU")

# Opening source image to work on
source_img = Image.open(source_img_filename)
# Convert it to a a properly sized and centered image into a tensor
x_img = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])(source_img)

_ = classify(x_img, correct_class = 281, normalize=True, show_image=False)

# Add dimension for NN training
x_img = x_img.unsqueeze(0)

# Create new output image initialized with existing image
x_hat = copy.deepcopy(x_img)

if is_cuda:
    x_hat = x_hat.cuda()
x_hat = Variable(x_hat)
x_hat.requires_grad = True

# GD step
y_hat_target = torch.LongTensor(1)
y_hat_target[0] = target_img_class
y_hat_target = Variable(y_hat_target)
if is_cuda:
    y_hat_target = y_hat_target.cuda()

criterions = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = [x_hat], lr = learning_rate_gd, momentum = momentum)

#inception_net.train()

print("Starting adversarial training...")

for i in range(steps):
    # zero the gradients... 
    optimizer.zero_grad()
    # Forward pass (calculate y_hat)
    y_hat = inception_net.forward(x_hat)
    # CAlculate loss y_hat vs y_hat_target
    loss = criterions(y_hat, y_hat_target)
    # Calculate gradients
    loss.backward()
    # Clip gradients
    x_hat.grad.data.clamp_(-epsilon, epsilon) #for p in x_hat.data.parameters():
        # p.grad.clamp_(-epsilon, epsilon)
        # _ = torch.nn.utils.clip_grad_norm(x_hat.grad, epsilon, norm_type=)
    # Update parameters with clipped gradients (h_hat in this case)
    optimizer.step()
    # Print progress
    if (i+1) % 10 == 0:
        print("Progress : pass " + str(i+1) + "/" + str(steps) + "; loss = " + str(loss.data[0]))
        #display_tensor(x_hat.data.cpu().squeeze(), denormalize_before_display=False)
    if loss.data[0] < loss_threshold:
        print("Loss below threshold of " + str(loss_threshold) + " reached at epoch " + str(i))
        break

print("Done training...!")

# _ = classify(newImage, correct_class = img_class, target_class= target_img_class, show_image=True, normalize= False)
_ = classify(x_hat.squeeze().data.cpu(), correct_class = img_class, target_class= target_img_class, show_image=True, normalize= False)
    
