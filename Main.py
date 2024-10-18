import torch.nn as nn
import torch
from torchvision import transforms, datasets
import sys

class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # depth wise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp), #(kernel_size, stride, padding)
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                # point wise
                nn.Conv2d(inp, oup, 1, 1, 0), #(kernel_size, stride, padding)
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )
            
        
        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),      #(224, 224, 3) -> (112,112,32)   
            conv_dw(32, 64, 1),         #(112,112,32) -> (112,112,64)
            conv_dw(64, 128, 2),        #(112,112,64) -> (56,56,128)
            conv_dw(128, 128, 1),       #(56,56,128) -> (56,56,128)
            conv_dw(128, 256, 2),       #(56,56,128) -> (28,28,256)
            conv_dw(256, 256, 1),       #(28,28,256) -> (28,28,256)
            conv_dw(256, 512, 2),       #(28,28,256) -> (14,14,512) 
            conv_dw(512, 512, 1),       #(14,14,512) -> (14,14,512)
            conv_dw(512, 512, 1),       #(14,14,512) -> (14,14,512) 
            conv_dw(512, 512, 1),       #(14,14,512) -> (14,14,512) 
            conv_dw(512, 512, 1),       #(14,14,512) -> (14,14,512) 
            conv_dw(512, 512, 1),       #(14,14,512) -> (14,14,512) 
            conv_dw(512, 1024, 2),      #(14,14,512) -> (7,7,1024)
            conv_dw(1024, 1024, 1),     #(7,7,1024) -> (7,7,1024)
            nn.AvgPool2d(7, stride=1)   # average (7,7)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)    #flatten into 1024 values then put into FC
        x = self.fc(x)
        return x

def guess():
    
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((224,224)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    img_dir = r"Img"
    
    img_data = datasets.ImageFolder(img_dir,transform = transform)
    
    img_loader = torch.utils.data.DataLoader(img_data)
      
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MobileNetV1(3,5).to(device)
    checkpoint = torch.load('checkpoint28.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_labels = ['PLANE','BIRD','CAR','CAT','DOG']
    with torch.no_grad():
        for image, _ in img_loader:
            image = image.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            del image, outputs
            return(class_labels[predicted])
        
import pygame
import matplotlib.pyplot as plt

pygame.init()

H = 400
W = 600

screen = pygame.display.set_mode([W,H])
screen.fill('black')

pygame.display.set_caption('Paint!')

drawing = False
last_pos = None

mouse_position = (0,0)

font = pygame.font.Font(r'C:\Windows\Fonts\ARLRDBD.ttf', 20)
guess_text = font.render('THIS IS A ...', True, (226, 241, 231),'black')
guessing = False


def give_answer():
    font2 = pygame.font.Font(r'C:\Windows\Fonts\ARLRDBD.ttf', 35)
    answer = font2.render('{}'.format(guess()), True, (98, 149, 132),'black')
    screen.blit(answer, (465,200))

while True:
    
    active_size = 5
    active_color = 'white'           
    
    #Guess bar
    pygame.draw.rect(screen, (36, 54, 66),[400,0,400,400])
    screen.blit(guess_text, (450,150))
    
    if guessing:
        give_answer()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            if (drawing):
                mouse_position = pygame.mouse.get_pos()
                if last_pos is not None:
                    if mouse_position[0]<400:
                        pygame.draw.line(screen, 'white', last_pos, mouse_position, active_size)
                    
                last_pos = mouse_position
                
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_position = (0, 0)
            drawing = False
            last_pos = None
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            
            if event.button == 1:
                drawing = True
                guessing = False
                
            elif event.button == 3:
                guessing = False
                screen.fill('black')
                
            elif event.button == 2:
                rect = pygame.Rect(0, 0, 400, 400)
                crop = screen.subsurface(rect)
                pygame.image.save(crop, r"Img/This folder/painting.png") 
                print("Processing")
                guessing = True
    
    pygame.display.update()

pygame.quit()