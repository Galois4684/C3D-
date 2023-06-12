import torch
from model import C3D
from dataset import C3DDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

num_classes = 2
lr = 1e-3
resume = 0
epochs = 5

writer = SummaryWriter(log_dir='logs')

model = C3D(num_classes=num_classes, weight_path='weights/pretrained_weight_without_fc8/c3d-pretrained.pth').cuda()

train_dataset = C3DDataset(root_dir='your_data/processed_data/train', num_frames=16)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

val_dataset = C3DDataset(root_dir='your_data/processed_data/val', num_frames=16)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.fc8.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    for epoch in range(resume, epochs):
        model.train()

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc8.parameters():
            param.requires_grad = True

        for batch_idx, batch_data in tqdm(enumerate(train_dataloader)):
            inputs, labels = batch_data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

        total_loss = 0
        model.eval()
        for batch_idx, batch_data in enumerate(val_dataloader):
            inputs, labels = batch_data
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            writer.add_scalar('Validation loss', loss.item(), epoch * len(val_dataloader) + batch_idx)

        avg_val_loss = total_loss / len(val_dataloader)
        writer.add_scalar('Average Validation Loss', avg_val_loss, epoch)

        torch.save(model.state_dict(), f'weights/model_epoch_{epoch}.pth')




