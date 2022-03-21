
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:\\Users\\misar\\Dropbox\\Mon PC (DESKTOP-HUVBVN6)\\Desktop\\Data\\RegressionDataset\\student_scores.csv")



def simpleLinerRegression(x,y):
    xi=x
    yi=y
    # moyen of the input
    x_mean = xi.mean()
    y_mean = yi.mean()
    # the total nbr of x element
    n = len(xi)
    
    # starte calculiting equation y = a + W*x
    Y1 = (xi*yi).sum()
    Y2 = (xi.sum()*yi.sum())/n
    
    X1 = (xi*xi).sum()
    X2 = (xi.sum()*xi.sum())/n
    
    # calculiting the Weight (W)
    w = (Y1-Y2)/(X1-X2)
    
    # calculeting intercept
    a = y_mean - w*x_mean
    return a,w

lenght = len(dataset)
train = dataset[:(int(lenght*0.8))]
test=dataset[(int(lenght*0.8)):]
print("x shape",train.shape,"y shape",test.shape)

a,w = simpleLinerRegression(train['Hours'],train['Scores'])
print("Y = ",a,"X*",w)

def gradianDesend(x,y,w,b,iteration,alpha):
    dw=0
    db=0
    m=x.shape[0]
    for i in range(iteration):
        for j in range(m):
            w = w-alpha*(dw/m)
            b = b-alpha*(db/m)
    return w,b