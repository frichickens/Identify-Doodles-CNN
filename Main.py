import torch.nn as nn
import torch
from torchvision import transforms, datasets

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

import pygame

pygame.init()

H = 254
W = 254
fps = 50000

timer = pygame.time.Clock()
screen = pygame.display.set_mode([H,W])

pygame.display.set_caption('Paint!')

run = True

painting = []
remove_list = []

def guess():
    
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize((224,224)),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    img_dir = r"C:\Users\baolo\Desktop\Lab\Test Project\SCs"
    
    img_data = datasets.ImageFolder(img_dir,transform = transform)
    
    
    img_loader = torch.utils.data.DataLoader(img_data,
                                             batch_size=1
                                             )
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MobileNetV1(3,5).to(device)
    checkpoint = torch.load('Checkpointv18.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_labels = ['airplane','bird','car','cat','dog']
    with torch.no_grad():
        for image, _ in img_loader:
            image = image.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            print(class_labels[predicted])
            print(outputs)
            del image, outputs
        
        
def draw(paints):
    for i in range(len(paints)):
        pygame.draw.circle(screen, paints[i][0], paints[i][1], paints[i][2])


while run:
    timer.tick(fps)
    screen.fill('white')
    mouse = pygame.mouse.get_pos()
    
    left_click = pygame.mouse.get_pressed()[0]
    middle_click = pygame.mouse.get_pressed()[1]
    right_click = pygame.mouse.get_pressed()[2]
    
    active_size = 1
    active_color = 'black'           
    
    pygame.draw.circle(screen, active_color, mouse, active_size)
    
    if left_click:
        painting.append((active_color, mouse, active_size))
        
    if right_click:
        # active_size = 15
        # active_color = 'white'    
        # painting.append((active_color, mouse, active_size))
        painting= []
    
    
    draw(painting)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            
            if event.button == 2: # 2 == middle button
                pygame.image.save(screen,"SCs/img/screenshot.png")
                print("Processing")
                guess()
    
    pygame.display.flip()
    timer.tick(fps)
    
pygame.quit()