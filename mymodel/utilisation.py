import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

#On transforme notre image en 32x32
transform = transforms.Compose([            
    transforms.Resize(32),                   
    transforms.CenterCrop(32),              
    transforms.ToTensor(),                 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#On récupère l'image
img = Image.open("cat.jpg")

#On transforme cette image en tensor
transformedImg = transform(img)
resizedImg = torch.unsqueeze(transformedImg, 0)

#classes connus par notre cnn
classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#On recrée un cnn en accord avec notre modèle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#Recharge le modèle
PATH = 'model.pth'
net.load_state_dict(torch.load(PATH))

#notre cnn va étudier l'image
outputs = net(resizedImg)

#et affiche sa prédiction
_, predicted = torch.max(outputs, 1)
print("Prédiction sur l'image".join('%5s' % classes[predicted[j]] for j in range(1)))

