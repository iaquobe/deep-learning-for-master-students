from skimage.io import imread
from PIL import Image
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import Resize
import torch 

weights    = MobileNet_V3_Small_Weights.DEFAULT
preprocess = weights.transforms()
model      = mobilenet_v3_small(weights=weights)



# rgb images
img = Image.open("./data/EuroSAT_RGB/River/River_1.jpg")
img = preprocess(img)


# multispectral images
img_2 = imread('./data/EuroSAT_MS/River/River_1.tif')

import numpy as np
img = (img_2[:,:, [3,2,1]] / 65535 * 255).astype(np.uint8)


Image.fromarray(img, mode="RGB")



img_2.shape

img_2 = torch.tensor(img_2).permute(2, 0, 1)
img_2 = Resize((224, 224))(img_2)
img_2 = img_2.float() / 65535.0
img_2 = img_2[[1, 2, 3, 4, 7, 12]]

img_2.shape


x.shape


x  = torch.concat([img, img]).unsqueeze(0)


img_2 = img_2.unsqueeze(0)



ms = Multispectral()



y  = ms(img_2)

y.shape


ms.parameters()

class Multispectral(torch.nn.Module): 
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

