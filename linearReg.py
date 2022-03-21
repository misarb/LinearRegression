
import matplotlib.pyplot as plt
import pandas as pd


#path of dataset
dataset = pd.read_csv("C:\\Users\\misar\\Dropbox\\Mon PC (DESKTOP-HUVBVN6)\\Desktop\\Data\\RegressionDataset\\student_scores.csv")



# def simpleLinerRegression(x,y):
#     xi=x
#     yi=y
#     # moyen of the input
#     x_mean = xi.mean()
#     y_mean = yi.mean()
#     # the total nbr of x element
#     n = len(xi)
    
#     # starte calculiting equation y = a + W*x
#     Y1 = (xi*yi).sum()
#     Y2 = (xi.sum()*yi.sum())/n
    
#     X1 = (xi*xi).sum()
#     X2 = (xi.sum()*xi.sum())/n
    
#     # calculiting the Weight (W)
#     w = (Y1-Y2)/(X1-X2)
    
#     # calculeting intercept
#     a = y_mean - w*x_mean
#     return a,w

lenght = len(dataset)
train = dataset[:(int(lenght*0.8))]
test=dataset[(int(lenght*0.8)):]
print("train_shape= ",train.shape,"test_shape = ",test.shape)

# a,w = simpleLinerRegression(train['Hours'],train['Scores'])




# model with gradient Desend
def model(x,y,iteration,learnigRate):
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
    #GradientDesend Algorithm
    for i in range(iteration):
        dw=0
        db=0
        for j in range(n):
            w = w-learnigRate*(dw/n)
            a=a-learnigRate*(db/n)
    #End Gradient Desend Algorithm


    return w,a

w,a = model(train['Hours'],train['Scores'],100,0.0000001)
print("Y = ",a,"X*",w)

#predact values
def predict_Regression(w,a,inputFeateur):
    predict = a+w*inputFeateur
    return predict

listInput = [1,80,15,20,50,13]

for i in range(len(listInput)):
    predacValue = predict_Regression(w,a,listInput[i])
    print("valueInput = ",listInput[i],"predacted Score = ",predacValue)