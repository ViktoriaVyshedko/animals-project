import torch


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
