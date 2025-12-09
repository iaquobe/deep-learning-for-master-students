from itertools import chain
import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MSConcatModel(torch.nn.Module): 
    '''Based on one mobilenet model. 
    
    For an input x with shape: (6, 224, 224)
    The model is run twice. Once on the first 3 channels, then on the last 3 channels. 
    The last classification layer of mobilenet is removed. 
    Instead the final layer brings together both parts of the input (2048 -> 10)

    Both parts of the input use the same weights for the cnn part
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        weights     = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        self.model.classifier.pop(3)

        self.final = torch.nn.Linear(2048, 10)


    def forward(self, x): 
        y1 = self.model(x[:, :3])
        y2 = self.model(x[:, 3:])
        y  = self.final(torch.concat([y1, y2], dim=-1))

        return y


    def get_parameters(self): 
        return chain(self.model.classifier.parameters(), self.final.parameters())


class MSAddModel(torch.nn.Module): 
    '''Based on two mobilenet model. 
    
    For an input x with shape: (6, 224, 224)
    The model is run twice. Once on the first 3 channels, then on the last 3 channels. 
    Both outputs are then added together in the classification layer.  
    Instead the final layer brings together both parts of the input (1024 -> 10)

    Here both parts of the input use different weights for the cnn part
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        weights     = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)
        self.model.classifier.pop(3)

        self.final = torch.nn.Linear(1024, 10)


    def forward(self, x): 
        y1 = self.model(x[:, :3])
        y2 = self.model(x[:, 3:])
        y  = self.final(y1 + y2)

        return y

    def get_parameters(self): 
        return chain(self.model.classifier.parameters(), self.final.parameters())

