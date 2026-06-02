# import torch
# import torch.nn as nn
# from torchvision.models import vgg19


# class PerceptualLoss(nn.Module):

#     def __init__(self):
#         super().__init__()
        
#         vgg = vgg19(weights="DEFAULT").features[:35]
#         self.feature_extractor = vgg.eval()
        
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
#         self.criterion = nn.L1Loss()
        
#         # CRITICAL FIX: Load to CUDA ONCE in __init__
#         # NOT in forward() - this was causing GPU stalls
#         self.feature_extractor = self.feature_extractor.cuda()
#         self.criterion = self.criterion.cuda()

#     def forward(self, prediction, target):
        
#         pred_features = self.feature_extractor(prediction)

#         with torch.no_grad():
#             target_features = self.feature_extractor(target)

#         loss = self.criterion(pred_features, target_features)

#         return loss

import torch
import torch.nn as nn
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():

            self.device = torch.device("cuda")

        elif torch.backends.mps.is_available():

            self.device = torch.device("mps")

        else:

            self.device = torch.device("cpu")

        vgg = vgg19(weights="DEFAULT").features[:35]

        self.feature_extractor = vgg.eval()

        for param in self.feature_extractor.parameters():

            param.requires_grad = False

        self.criterion = nn.L1Loss()

        self.feature_extractor = self.feature_extractor.to(
            self.device
        )

        self.criterion = self.criterion.to(
            self.device
        )

    def forward(
        self,
        prediction,
        target
    ):

        prediction = prediction.to(
            self.device
        )

        target = target.to(
            self.device
        )

        pred_features = self.feature_extractor(
            prediction
        )

        with torch.no_grad():

            target_features = self.feature_extractor(
                target
            )

        loss = self.criterion(

            pred_features,

            target_features

        )

        return loss