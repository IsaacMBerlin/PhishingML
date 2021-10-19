import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch import nn, optim
from torch.autograd import Variable

torch.Tensor.ndim = property(lambda self: len(self.shape))
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#fixing the data
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data, labels):
        self.labels = labels
        self.data = data

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        x = self.data[index]
        x = x.astype(np.float32)
        y = self.labels[index]

        return x, y

PreData = pd.read_csv("combined_dataset.csv")
del PreData["domain"]

data_df = PreData.iloc[:,:-1]
label_df = PreData.iloc[:,-1]

d = preprocessing.normalize(data_df, copy=False)

full_dataset = MyDataset(d, label_df)

def load_split_train_test(full_dataset, valid_size = .5):
    num_train = len(full_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(full_dataset, sampler=train_sampler, batch_size=512)
    testloader = torch.utils.data.DataLoader(full_dataset, sampler=test_sampler, batch_size=512)
    return trainloader, testloader

train_dataloader, test_dataloader = load_split_train_test(full_dataset)

#trainDataLoader = torch.utils.data.DataLoader(train_datasetet, batch_size=512)
#testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

#the neural net
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self). __init__()
        self.linear1 = nn.Linear(10, 250)
        self.linear2 = nn.Linear(250, 50)
        self.final_linear = nn.Linear(50,2)

        self.relu = nn.ReLU()

    def forward(self, images):
        x = images.view(-1, 10)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.final_linear(x))
        return x

#process the data
model = Mnet()
model.train()
cec_loss = nn.CrossEntropyLoss()
params = model.parameters()
optimiser = optim.SGD(params=params, lr=0.1, momentum= 0.9)
plotIterTrain = []
plotLossTrain = []
plotIterTest = []
plotLossTest = []

n_epochs=50

for e in range(n_epochs):
    print("e=",e)
    train_loss = 0
    for i, (info, labels) in enumerate (train_dataloader):

        print("i=",i)
        info = Variable(info)
        labels = Variable(labels)
        output = model(info)

        model.zero_grad()
        loss = cec_loss(output,labels)
        train_loss += loss.item()

        print(loss.item())
        loss.backward()

        optimiser.step()

    plotIterTrain.append(e)
    plotLossTrain.append(loss.item())

    print("testing")
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            output = model.forward(inputs)
            loss = cec_loss(output, labels)
            print(loss.item())
            test_loss += loss.item()
    model.train()
    plotIterTest.append(e)
    plotLossTest.append(test_loss)


#graph the data
plt.plot(plotIterTrain, plotLossTrain)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Progress over Time")
plt.savefig("mygraph_train.png")
plt.cla()

plt.plot(plotIterTest, plotLossTest)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Test Progress over Time")
plt.savefig("mygraph_test.png")
