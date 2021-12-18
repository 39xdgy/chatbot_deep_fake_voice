# -*- coding: utf-8 -*-

import torch
import os
from IPython.display import Audio
from PIL import Image
import matplotlib.pyplot as plt
import copy


def download_from_url_ffmpeg(url, output, minute_mark=1):
    try:
        os.remove(output)
    except:
        pass

    cmd = 'ffmpeg -loglevel warning -ss 0 -i $(youtube-dl -f bestaudio --get-url ' + str(url) + ') -t ' + str(
        minute_mark * 120) + ' ' + str(output)
    os.system(cmd)

    return os.getcwd() + '/' + output

# Modify the voice resource here
# url1 = 'https://www.youtube.com/watch?v=WE6mnPmztoQ'
# content_audio_name = download_from_url_ffmpeg(url1, 'origin.wav')
# url2 = 'https://www.youtube.com/watch?v=B8jUVci17vE'
# style_audio_name = download_from_url_ffmpeg(url2, 'Style.wav')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, AvgPool1d, MaxPool2d, Linear, Conv1d
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms
import copy
import librosa


class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c = input.size()
        features = input.view(a * b, c)
        G = torch.mm(features, features.t())
        return G.div(a * b * c)


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


import gc;

gc.collect()

N_FFT = 2048


def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, fs


style_audio, style_sr = read_audio_spectum(style_audio_name)
content_audio, content_sr = read_audio_spectum(content_audio_name)

if (content_sr != style_sr):
    print('Sampling rates are not same')

style_audio = style_audio.reshape([1, 1025, style_audio.shape[1]])
content_audio = content_audio.reshape([1, 1025, content_audio.shape[1]])

if torch.cuda.is_available():
    style_float = Variable((torch.from_numpy(style_audio)).cuda())
    content_float = Variable((torch.from_numpy(content_audio)).cuda())
else:
    style_float = Variable(torch.from_numpy(style_audio))
    content_float = Variable(torch.from_numpy(content_audio))


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.cnn2 = Conv1d(in_channels=4096, out_channels=4096, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu(out)
        out = self.cnn2(x)
        out = self.relu(out)
        out = self.cnn3(x)
        return out


cnn = CNNModel()
if torch.cuda.is_available():
    cnn = cnn.cuda()

style_weight = 1000
content_weight = 2


def get_style_model_and_losses(cnn, style_float, \
                               content_float=content_float, \
                               style_weight=style_weight):
    cnn = copy.deepcopy(cnn)

    style_losses = []
    content_losses = []

    model = nn.Sequential()

    gram = GramMatrix()

    if torch.cuda.is_available():
        model = model.cuda()
        gram = gram.cuda()

    name = 'conv_1'
    model.add_module(name, cnn.cnn1)

    name = 'relu1'
    model.add_module(name, cnn.relu)

    name = 'conv_2'
    model.add_module(name, cnn.cnn2)

    target_feature = model(style_float).clone()
    target_feature_gram = gram(target_feature)
    style_loss = StyleLoss(target_feature_gram, style_weight)
    model.add_module("style_loss_1", style_loss)
    style_losses.append(style_loss)

    target = model(content_float).detach()
    content_loss = ContentLoss(target, content_weight)
    model.add_module("content_loss_1", content_loss)
    content_losses.append(content_loss)

    return model, style_losses, content_losses


get_style_model_and_losses(cnn, style_float, content_float)

import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, AvgPool1d, MaxPool2d, Linear, Conv1d
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms

import gc;

gc.collect()

input_float = content_float.clone()

learning_rate_initial = 0.0005


def get_input_param_optimizer(input_float):
    input_param = nn.Parameter(input_float.data)
    optimizer = optim.Adam([input_param], lr=learning_rate_initial)
    return input_param, optimizer


num_steps = 5000


def run_style_transfer(cnn, style_float=style_float, \
                       content_float=content_float, \
                       input_float=input_float, \
                       num_steps=num_steps, style_weight=style_weight):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_float, content_float)
    input_param, optimizer = get_input_param_optimizer(input_float)
    print('Optimizing..')
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data


output = run_style_transfer(cnn, style_float=style_float, content_float=content_float, input_float=input_float)

if torch.cuda.is_available():
    output = output.cpu()

output = output.squeeze(0)
output = output.numpy()

N_FFT = 2048
a = np.zeros_like(output)
a = np.exp(output) - 1

p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(1000):
    S = a * np.exp(1j * p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

import soundfile as sf

OUTPUT_FILENAME = 'output.wav'
sf.write(OUTPUT_FILENAME, x, style_sr)

import noisereduce as nr
from scipy.io import wavfile
# noise cancel
rate, data = wavfile.read("output.wav")
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("final.wav", rate, reduced_noise)

