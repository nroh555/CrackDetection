import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset``
import albumentations as A
import random
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpUtils

#Define variables
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ['crack']
ACTIVATION ='sigmoid'
DEVICE = 'cuda'

#Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

train = pd.read_csv(r"C:\Users\naren\OneDrive\Desktop\data-120\train_120.csv")
validation = pd.read_csv(r"C:\Users\naren\OneDrive\Desktop\data-120\validation_120.csv")
test = pd.read_csv(r"C:\Users\naren\OneDrive\Desktop\data-120\test_120.csv")

#Subclass dataset to create a training set
class SegDataTrain(Dataset):
    def __init__(self,df,classes=None,transform=None):
        self.df = df
        self.classes = classes
        self.transform = transform
    
    def __getitem__ (self,idx):
        image_name = self.df.iloc[idx,1]
        mask_name = self.df.iloc[idx,2]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name,cv2.IMREAD_UNCHANGED)
        image = image.astype('uint8')
        mask = mask[:,:,0]
        if(self.transform is not None):
            random.seed(42),
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2,0,1)
            image = image.float()/255
            mask = mask.long().unsqueeze(0)
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2,0,1)
            image = image.float()/255
            mask = mask.long().unsqueeze(0)
        return image,mask
    
    def __len__ (self):
        return len(self.df)

#Define transforms using albumations
validation_transform = A.Compose([A.Resize(512,512)])
train_transform = A.Compose([A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),])
#Create the datasets
trainDS = SegDataTrain(train, classes = CLASSES, transform = train_transform)
valDS = SegDataTrain(validation, classes = CLASSES, transform = validation_transform)

#Define DataLoaders  
trainDL = torch.utils.data.DataLoader(trainDS, batch_size = 4, shuffle=True, sampler=None,
    batch_sampler=None, num_workers=0, collate_fn=None,
    pin_memory=False, drop_last=False, timeout=0,
    worker_init_fn=None)

valDL =  torch.utils.data.DataLoader(valDS, batch_size= 4, shuffle=False, sampler=None,
    batch_sampler=None, num_workers=0, collate_fn=None,
    pin_memory=False, drop_last=False, timeout=0,
    worker_init_fn=None)

#Check Tensor shapes
batch = next(iter(trainDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

#Check first sample from patch
testImg = images[1]
testMsk = labels[1]
print(testImg.shape, testImg.dtype, type(testImg), testMsk.shape, 
        testMsk.dtype, type(testMsk), testImg.min(), 
        testImg.max(), testMsk.min(), testMsk.max())

# Plot example image =====================================
plt.imshow(testImg.permute(1,2,0))
plt.show()
plt.close()

# Plot exmaple mask ======================================
plt.imshow(testMsk.permute(1,2,0))
plt.show()
plt.close()

#Initiate the Unet++ model
model = smp.UnetPlusPlus(   
    encoder_name = ENCODER,
    encoder_weights = ENCODER_WEIGHTS,
    in_channels = 3,
    classes = 2,
    activation = ACTIVATION
)

#Define Loss and Metrics to Monitor ======================================
loss = smpUtils.losses.DiceLoss()
metrics = [smpUtils.metrics.Fscore(eps=1e-7, threshold=0.5, activation = ACTIVATION, ignore_channels=None),
            smpUtils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None), 
            smpUtils.metrics.Recall(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None), 
            smpUtils.metrics.Precision(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None)]

# Define Optimizer (Adam in this case) and learning rate ============================
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

# Define training epock =====================================
train_epoch = smpUtils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device= DEVICE,
    verbose = True
)

# Define testing epoch =====================================
val_epoch = smpUtils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device= DEVICE,
    verbose = True
)

# Train model for 100 epochs ==================================
max_score = 0

for i in range(1, 51):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    test_logs = val_epoch.run(valDL)

    # do something (save model, change lr, etc.)
    if max_score < test_logs['fscore']:
        max_score = test_logs['fscore']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# Load saved model ============================================
best_model = torch.load('./best_model.pth')

testDS = SegDataTrain(test, classes=CLASSES, transform=validation_transform)

testDL =  torch.utils.data.DataLoader(testDS, batch_size=4, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(testDL)
print(logs)

#Visualize masks, and predictions=======================================
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

for i in range(10):
    n = np.random.choice(len(testDS))
    
    image_vis = testDS[n][0].permute(1,2,0)
    image_vis = image_vis.numpy()*255
    image_vis = image_vis.astype('uint8')
    image, gt_mask = testDS[n]

    gt_mask = gt_mask.squeeze()

    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    m = nn.Softmax(dim=1)
    pr_probs = m(pr_mask)              
    pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask, 
        predict_mask = pr_mask,
    )