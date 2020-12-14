import torchvision
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, backward
import matplotlib.pyplot as plt
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

        self.dropout=dropout
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

if __name__ == "__main__":
    transform = torchvision.transforms.ToTensor()
    mnist_path = 'data/'
    mnist_train = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.MNIST(root=mnist_path, train=False, transform=transform)
    input_dim = 784
    output_dim = 10
    # model = LogisticRegression(input_dim=input_dim, output_dim=output_dim)
    model = Mlp(n_classes=10, dropout=False)

    batch_size = 128
    no_epochs = 5

    optimizer = Sls(params=model.parameters(), c=0.1)
    val_size = int(len(mnist_train) * 0.1)
    train_size = len(mnist_train) - val_size

    training_errors = []
    validation_errors = []

    criterion = nn.CrossEntropyLoss(reduction="mean")

    for epoch in range(200):
        model.train()
        train_set, val_set = torch.utils.data.random_split(mnist_train, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        for (images, labels) in train_loader:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(images)
            optimizer.step(lambda: criterion(outputs, labels.view(-1)))
        
        correct = 0
        for (images, labels) in train_loader:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            outputs = model(images)
            pred = torch.nn.functional.softmax(outputs, dim=1)
            for i, p in enumerate(pred):
                if labels[i] == torch.max(p.data, 0)[1]:
                    correct = correct + 1
        training_error = 1 - correct / train_size
        training_errors.append(training_error)

        val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
        model.eval()
        correct = []
        for (images, labels) in val_loader:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)
            outputs = model(images)
            pred_labels = outputs.argmax(dim=1)
            acc = (pred_labels == labels).float().mean()
            correct.append(acc)
            # pred = torch.nn.functional.softmax(outputs, dim=1)
        #     print(pred)
        #     for i, p in enumerate(pred):
        #         # print(torch.argmax(p))
        #         if labels[i] == torch.argmax(p):
        #             correct = correct + 1
        # valid_error = 1 - correct / val_size
        valid_acc = np.sum(np.array(correct)) / len(correct)
        validation_errors.append(valid_acc)
        print('Epoch {}/{} with training error {:.6f} and validation acc {:.6f}'.format(epoch+1, no_epochs, training_error, valid_acc))


    plt.xlabel('Iterations')
    plt.plot(training_errors, label='training error')
    plt.plot(validation_errors, label='validation error')
    plt.legend()
    plt.show()
