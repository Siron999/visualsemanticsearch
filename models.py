from typing import Union
from PIL import Image
import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer


class ResNetFeatureExtraction(nn.Module):
    def __init__(self, pre_trained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pre_trained else None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        resnet = models.resnet50(weights=weights)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: Union[torch.Tensor, Image.Image]):
        if isinstance(x, Image.Image):
            x = self.transform(x).unsqueeze(0)
        features = self.features(x)
        return features.view(features.size(0), -1).squeeze(0)


class BERTEmbeddings(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, text):
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return self.model.encode(text)
