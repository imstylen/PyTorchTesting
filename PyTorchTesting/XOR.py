

from __future__ import print_function
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

        self.fc1 = nn.Linear(2,20)
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

def train(args, model, device, X,Y, optimizer, epoch):

    model.train()

    X = X.to(device)
    Y = Y.to(device)

    optimizer.zero_grad()
    output = model(X)
    loss = F.mse_loss(Y,output)
    loss.backward();
    optimizer.step();

    if(epoch % args.log_interval == 0):
        print('batch loss: {}'.format(loss.item()))



def main():

    ########################################################################################
    ## Parser (Training settings)
    ########################################################################################
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #batch size:
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    #test batch size:
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    #epoch:
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    #learning rate:
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    #gamma:
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    #no-cuda:
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #seed:
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #log interval:
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    #save model:
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(torch.cuda.is_available())
    ########################################################################################
    # Load data
    ########################################################################################
    

    X = torch.tensor([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
    Y = torch.tensor([[0.0],[1.0],[1.0],[0.0]])

    ########################################################################################
    # Create Model
    ########################################################################################

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    ########################################################################################
    # Train Model
    ########################################################################################

    for epoch in range(1, args.epochs + 1):
        for i in range(0,4):
            train(args, model, device, X[i],Y[i], optimizer, epoch)

    ########################################################################################
    # Save Model
    ########################################################################################

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

########################################################################################
# Main
########################################################################################

if __name__ == '__main__':
    main()