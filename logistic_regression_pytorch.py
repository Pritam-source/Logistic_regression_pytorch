import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bc=datasets.load_breast_cancer()
X,y= bc.data,bc.target
n_samples,n_features=X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
    def forward(self,x):   
        y_predicted=torch.sigmoid(self.linear(x))
        return y_predicted

model=LogisticRegression(n_features)
loss_fn=nn.BCELoss()
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
n_iters=100
for epoch in range(n_iters):
    y_pred=model(X_train)
    loss=loss_fn(y_pred,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 ==0:
        [w,b]=model.parameters()
        print(f'epoch:{epoch+1},W={w[0][0].item():.3f},loss={loss.item():.4f}')


with torch.no_grad():
    y_predicted=model(X_test)
    y_predicted_cls=y_predicted.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy={acc:.4f}')

    
