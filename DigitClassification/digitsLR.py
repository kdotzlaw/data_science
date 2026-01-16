import torch
import torch.nn as nn
import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.autograd import Variable

'''
Create logistic regression class

'''
class LogisticRegression(nn.Module):
    # constructor
    def __init__(self, inputSize, numClasses):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(inputSize,numClasses)
    # caclculate softmax with forward pass
    def forward(self, x):
        return self.linear(x)



if __name__=='__main__':
    # download  MNIST handwritten digits dataset into memory
    batchSize = 100
    print("Downloading...")
    training = ds.MNIST(root='./data', train=True,transform=transforms.ToTensor(),download=True)
    testing = ds.MNIST(root='./data',train=False,transform=transforms.ToTensor())
    # load dataset
    print("Loading...")
    trainLoad = torch.utils.data.DataLoader(dataset=training,batch_size=batchSize, shuffle=True)
    testLoad = torch.utils.data.DataLoader(dataset=testing,batch_size=batchSize,shuffle=False)

    # define hyperparameters
    inputSize = 784 # images 28x28 so 784px
    numClasses = 10 # 10 digits in dataset
    epochs = 5 # run training 5 times
    batchSize = 100 # train on 100 images each to prevent memory overflow
    learningRate = 0.001

    # create LR model
    model = LogisticRegression(inputSize,numClasses)

    # set loss function to use cross entropy
    crit = nn.CrossEntropyLoss()

    # set optimizer to use stochastic gradient descent with learningRate
    optimize = torch.optim.SGD(model.parameters(),lr=learningRate)
    
    # Train model
    for epoch in range(epochs):
        for i, (img,labels) in enumerate(trainLoad):
            
            # set images & labels
            img = Variable(img.view(-1,28*28))
            labels = Variable(labels)

            # reset gradients to 0
            optimize.zero_grad()
            # forward pass
            output = model(img)
            # calculate loss
            loss = crit(output,labels)
            # backpropegation
            loss.backward()
            # update weights
            optimize.step()
            
            # print out epochs and associated loss calculations
            if(i+1)%100==0:
                
                #print(f"Epoch: {epoch+1}/{epochs}, step: [{i+1}/{len(training)//batchSize}], Loss: {loss.item()}")
                print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
				% (epoch + 1, epochs, i + 1, 
					len(training) // batchSize, loss.item()))
    # test model
    correct = 0
    total = 0
    for img, labels in testLoad:
        img  = Variable(img.view(-1,28*28))
        output = model(img)
        # find predicted values
        _,pred = torch.max(output.data,1)
        # count total img processed
        total+=labels.size(0)
        # count correct predictions
        correct+=(pred==labels).sum()

    print(f"Model accuracy: {(correct/total)*100}")