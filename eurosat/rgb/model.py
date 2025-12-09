import torch 
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights

class RGBModel(torch.nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        weights                  = MobileNet_V3_Small_Weights.DEFAULT
        self.model               = mobilenet_v3_small(weights=weights)
        self.model.classifier[3] = torch.nn.Linear(1024, 10)

    def forward(self, x): 
        return self.model(x)
