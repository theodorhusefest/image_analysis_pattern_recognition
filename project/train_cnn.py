from CNN import OperatorsDataset, get_dataloaders, Classifier

import os
import time
import torch
from torchvision import transforms, utils

def train():
    
    epochs = 50
    eval_every = 1
    print_every = 1
    train_accs = []
    val_accs = []
    best_acc = 0
    
    transform = transforms.Compose([transforms.RandomRotation(180), 
                        transforms.RandomRotation(90),
                        transforms.ToTensor(), 
                        transforms.Normalize(0.1307, 0.3081)])

    train_dataset = OperatorsDataset("./data/train", "./data/datasheet.csv", transform = transform)
    
    
    SAVEDIR = "run_{}".format(np.around(time.time(), 2))
    os.makedirs(SAVEDIR)

    net = OperatorNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Currently using device: {}".format(device))
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.2, verbose = True, patience = 3, threshold=0.01)

    for e in range(epochs): 
        running_loss = 0
        start = time.time()
        for images, labels in iter(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            images = images.unsqueeze(1)

            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if e % eval_every == 0:

            accuracy_train = validate_on_dataloader(net, train_loader)
            accuracy_val = validate_on_dataloader(net, valid_loader)
            if accuracy_val > best_acc:
                best_acc = accuracy_val
                torch.save(net.state_dict(), "./{}/val_acc{}.ckpt".format(SAVEDIR, np.around(best_acc, 2)))

            train_accs.append(accuracy_train)
            val_accs.append(accuracy_val)
            if e % print_every == 0:
                stop = time.time()
                print("\n----------------------------------------")
                print("EPOCH {}".format(e))
                print("Epoch done in {}".format(np.around(stop - start, 2)))
                print("Loss on current epoch: {}".format(running_loss))
                print("Accuracy on train set {}".format(np.around(accuracy_train, 2)))
                print("Accuracy on validation set {}".format(np.around(accuracy_val, 2)))
                print("----------------------------------------")

        scheduler.step(running_loss)
