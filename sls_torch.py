import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from sls_optimizer import Sls
from torch.nn import functional as F



class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

     def forward(self, x):
        return self.linear(x)


class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.dropout = dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits


def softmax_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc


@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_function):  
    model.eval()
    loader = DataLoader(dataset, drop_last=False, batch_size=1024)

    score_sum = 0.
    for images, labels in loader:
        images, labels = images.cuda(), labels.cuda()
        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score


if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,))])
    mnist_path = 'data/'
    mnist_train = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
    mnist_train_loader = DataLoader(mnist_train, drop_last=True, shuffle=True, batch_size=128)
    mnist_test = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
    input_dim = 784
    output_dim = 10
    # model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    model = Mlp(n_classes=10, dropout=False).cuda()

    batch_size = 128
    no_epochs = 5

    n_batches_per_epoch = len(mnist_train)/128
    optimizer = Sls(params=model.parameters(), c=0.1, n_batches_per_epoch=n_batches_per_epoch)
    val_size = int(len(mnist_train) * 0.1)
    train_size = len(mnist_train) - val_size

    training_errors = []
    validation_errors = []

    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(200):
        train_loss = compute_metric_on_dataset(model, mnist_train, softmax_loss)
        val_acc = compute_metric_on_dataset(model, mnist_test, softmax_accuracy)

        print('epoch {} with training loss {:6f}, validation acc {:6f}, and step size {:6f}'.format(epoch, train_loss, val_acc, optimizer.state["step_size"]))

        model.train()
        for images,labels in mnist_train_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()

            closure = lambda : softmax_loss(model, images, labels, backwards=False)
            optimizer.step(closure)


    plt.xlabel('Iterations')
    plt.plot(training_errors, label='training error')
    plt.plot(validation_errors, label='validation error')
    plt.legend()
    plt.show()
