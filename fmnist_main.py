import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data as data
#import transforms
import numpy as np
from inception_densenet import InceptionV4
# from torchsummary import summary
# from model import CNN
from resnet import ResNet
#from densenet import DenseNet
#import densenet as dn
# Training hyperparameters
epochs = 300
batch_size = 16 #72 densenet
learning_rate = 0.01
momentum = 0.9
log_interval = 20

best_acc = 0
best_epoch = 0


def save_predictions(model, device, test_loader, path):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            with open(path, "a") as out_file:
                np.savetxt(out_file, output)


def predict_batch(model, device, test_loader):
    examples = enumerate(test_loader)
    model.eval()
    with torch.no_grad():
        batch_idx, (data, target) = next(examples)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        pred = pred.numpy()
    return data, target, pred


def plot_graph(train_x, train_y, test_x, test_y, ylabel=''):
    plt.plot(train_x, train_y, color='blue')
    plt.plot(test_x, test_y, color='red')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(ylabel + '.png', dpi=100)
    # plt.show()


def train(criterion, model, device, train_loader, optimizer, epoch, losses=[], counter=[], errors=[]):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print('O',data, output)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            losses.append(loss.item())
            counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    errors.append(100. * (1 - correct / len(train_loader.dataset)))


def test(criterion, model, device, test_loader, epoch, losses=[], errors=[]):
    global best_acc
    global best_epoch
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(data, output)

            test_loss += criterion(output, target).item()
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if acc > best_acc:
        best_acc = acc
        best_epoch = epochs
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    losses.append(test_loss)
    errors.append(100. * (1 - correct / len(test_loader.dataset)))


def main():
    use_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("CUDA", use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    # data transformation
    train_data = datasets.FashionMNIST('data', download=True, train=True,
                                       transform=transforms.Compose([
                                           transforms.RandomCrop(28, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           # transforms.Resize((299,299)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.2860,), (0.1261,))
                                       ]))
    test_data = datasets.FashionMNIST('data', download=True, train=False,
                                      transform=transforms.Compose([
                                          # transforms.Resize((299,299)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.2868,), (0.1258,))
                                      ]))


    # data loaders
    kwargs = {'num_workers': 4 * gpu_count, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    # extract and plot random samples of data
    examples = enumerate(test_loader)
    batch_idx, (data, target) = next(examples)
    # plot_data(data, target, 'Ground truth')

    # model creation
    model = torch.nn.DataParallel(InceptionV4(32, [5,10,5], 0.5)).to(device)
    #model = torch.nn.DataParallel(ResNet(50)).to(device)
    # summary(model, (1, 32, 32))
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # optimizer creation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96)

    # lists for saving history
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]
    train_errors = []
    test_errors = []
    error_counter = [i * len(train_loader.dataset) for i in range(epochs)]

    # test of randomly initialized model
    test(criterion, model, device, test_loader, 0, losses=test_losses)

    # global training and testing loop
    for epoch in range(1, epochs + 1):
        scheduler.step()
        train(criterion, model, device, train_loader, optimizer, epoch, losses=train_losses, counter=train_counter, errors=train_errors)
        test(criterion, model, device, test_loader, epoch, losses=test_losses, errors=test_errors)

    # plotting training history
    plot_graph(train_counter, train_losses, test_counter, test_losses, ylabel='negative log likelihood loss')
    plot_graph(error_counter, train_errors, error_counter, test_errors, ylabel='error (%)')

    # extract and plot random samples of data with predicted labels
    # data, _, pred = predict_batch(model, device, test_loader)
    # plot_data(data, pred, 'Predicted')
    torch.save(model.state_dict(), 'CNN.pt')

    save_predictions(model, device, test_loader, 'predictions.txt')


    print(best_acc, best_epoch)


if __name__ == '__main__':
    main()
