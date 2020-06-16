from __future__ import print_function
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


########################################################################################
# Define Neural Network
########################################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4,20)
        self.fc2 = nn.Linear(20,1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = torch.sigmoid(x)
        return output

########################################################################################
# Train Model
########################################################################################

def train(model, device, X,Y, optimizer, epoch, log_interval):

    model.train()

    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    output = model(X.float())
    loss = F.mse_loss(Y,output)
    loss.backward();
    optimizer.step();

    if(epoch % log_interval == 0):
        print('batch loss: {}'.format(loss.item()))



def main():

    epochs = 100
    log_interval = 10
    save_model = True
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    ########################################################################################
    # Load data
    ########################################################################################
    

    X = []
    Y = []

    for i in range(0,100):
        x = list()
        for j in range(0,4):
            x.append(random.randint(0,1))
        X.append(x)
        if x == [1,0,0,1] or x == [0,1,1,0]:
            Y.append([1])
        else:
            Y.append([0])


    X = torch.tensor(X)
    Y = torch.tensor(Y)

    ########################################################################################
    # Create Model
    ########################################################################################

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    ########################################################################################
    # Train Model
    ########################################################################################

    for epoch in range(1, epochs + 1):
        for i in range(0,X.size()[0]):
            train( model, device, X[i],Y[i], optimizer, epoch, log_interval)
    
    print(model(torch.tensor([0,1,1,0]).float()))
    print(model(torch.tensor([1,1,1,0]).float()))
    ########################################################################################
    # Save Model
    ########################################################################################

    if save_model:
        torch.save(model.state_dict(), "diagonal.pt")

########################################################################################
# Main
########################################################################################

if __name__ == '__main__':
    main()