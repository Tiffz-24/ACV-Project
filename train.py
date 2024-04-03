import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def train_network(net, trainloader, criterion=CrossEntropyLoss(), optimizer_class=Adam, n_epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    net.to(device)
    
    # Instantiate the criterion if it's a class.
    if isinstance(criterion, type):
        criterion = criterion()
    
    # Instantiate the optimizer with the network parameters.
    optimizer = optimizer_class(net.parameters())

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print("starting epoch ", epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print("got data")
            inputs, labels = inputs.to(device), labels.to(device)
            print("inputs and labels to device")

            # zero the parameter gradients
            optimizer.zero_grad()
            print("zero grad")
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print statistics
            if i % 1000 == 0 and i > 0:
                print(f'Epoch={epoch + 1} Iter={i + 1:5d} Loss={running_loss / 1000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return net
