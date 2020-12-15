import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sls_optimizer import Sls
from torch.nn import functional as F
import argparse


class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


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


def logistic_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '-dataset', required=True)
    args = parser.parse_args()

    epochs = 30

    if args.d is not None:
        if args.d == 'mushroom':
            shroom = fetch_openml('mushroom', version=1)
            X, y = shroom['data'], shroom['target']
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)

            labels = np.unique(y)

            y[y==labels[0]] = 0.
            y[y==labels[1]] = 1.

            y = y.astype('float64')

            splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
            X_train, X_test, Y_train, Y_test = splits

            X_train = torch.FloatTensor(X_train)
            Y_train = torch.FloatTensor(Y_train)
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.FloatTensor(Y_test)

            batch_size = 100

            dataset_train = torch.utils.data.TensorDataset(X_train, Y_train)
            dataset_train_loader = DataLoader(dataset_train, drop_last=True, shuffle=True, batch_size=batch_size)
            dataset_test = torch.utils.data.TensorDataset(X_test, Y_test)

            input_dim = 22
            output_dim = 1
            model = LinearRegression(input_dim=input_dim, output_dim=output_dim).cuda()

            metric_fun = logistic_loss
            val_fun = logistic_accuracy
            

        elif args.d == 'mnist':
            transform = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.5,), (0.5,))])
            mnist_path = 'data/'
            dataset_train = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
            dataset_train_loader = DataLoader(dataset_train, drop_last=True, shuffle=True, batch_size=128)
            dataset_test = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
            input_dim = 784
            output_dim = 10
            # model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
            model = Mlp(n_classes=10, dropout=False).cuda()

            batch_size = 128

            metric_fun = softmax_loss
            val_fun = softmax_accuracy
    
        else:
            raise ValueError("invalid datasets")

        n_batches_per_epoch = len(dataset_train)/batch_size
        optimizer = Sls(params=model.parameters(), c=0.1, n_batches_per_epoch=n_batches_per_epoch)

        accs = []
        alphas = []

        for epoch in range(30):
            train_loss = compute_metric_on_dataset(model, dataset_train, metric_fun)
            val_acc = compute_metric_on_dataset(model, dataset_test, val_fun)
            accs.append(val_acc)
            alphas.append(optimizer.state["step_size"])
            print('epoch {} with training loss {:6f}, validation acc {:6f}, and step size {:.5e}'.format(epoch, train_loss, val_acc, optimizer.state["step_size"]))

            model.train()
            for images,labels in dataset_train_loader:
                images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()

                closure = lambda : metric_fun(model, images, labels, backwards=False)
                optimizer.step(closure)

        plt.scatter(alphas[1:], accs[1:])
        plt.xlabel('learning rates')
        plt.ylabel('accuracy')
        plt.title('SLS ' + args.d + ' Learning Rate vs Accuracy')
        plt.show()
