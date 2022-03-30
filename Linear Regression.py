
import numpy as np

# Loading Training Dataset
X_data = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
Y_data = np.load("age_regression_ytr.npy")

# Splitting into Training and Validation Dataset
X_tr, X_va= np.split(X_data,[int(.8 * len(X_data))])
X_tr= X_tr.T
X_va= X_va.T
Y_tr, Y_va= np.split(Y_data, [int(.8 * len(X_data))])

# Loading Test Dataset
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
X_te= X_te.T
Y_te = np.load("age_regression_yte.npy")

# Number of Training Examples
N=np.shape(X_tr)[1]

# Hyper Parameter Values
# Uncomment below 4 lines and comment line 28-31 to run it for one Hyperparameter Set
# Mb=[32]
# Epoch=[750]
# Alpha=[1]
# Eps=[0.00075]
Mb=[2, 4, 5, 8, 10, 16, 20, 32, 40, 100] # Mini Batch Size
Epoch= [1, 10, 25, 50, 75 ,100, 250, 500, 750, 1000] # Number of Epochs
Alpha= [100, 10, 1 ,0.5, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001] # Regularization Strength
Eps= [0.0025, 0.001, 0.00075, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001] # Learning Rate

mincost=np.Inf
H_star=np.array(4) #Initializing an array to store Best Hyperparameters

print("----------------------------------------------------------------------")
print("Performing Grid Search for 10e4 combinations of Hyperparameters:")
print("----------------------------------------------------------------------")

W=np.random.randn(np.shape(X_tr)[0])*0.01  #Randomly intializing the Weights
B=0 #Initializing bias = 0

for epoch in Epoch:
    for alpha in Alpha:
        for eps in Eps:
            for mb in Mb:
                # Stochastic Gradient Descent
                # w=np.random.randn(np.shape(X_tr)[0])*0.01 #Randomly intializing the Weights
                # b=0 #Initializing bias = 0
                w=W
                b=B
                for i in range(epoch):
                    for j in range(int(N/mb)):
                        X_mb=X_tr[:,j*mb:(j+1)*mb]
                        Y_mb=Y_tr[j*mb:(j+1)*mb]

                        # Calculating the Gradients
                        dw= (np.dot(X_mb,(np.dot(X_mb.T,w)+b-Y_mb))+alpha*w)/mb
                        db= np.sum(np.dot(X_mb.T,w)+b-Y_mb)/mb

                        # Updating the Weights and Bias
                        w-=eps*dw
                        b-=eps*db

                # Calculating Regularized MSE Loss on Validation Set
                Y_va_hat= np.dot(X_va.T,w)+b
                cost= np.dot((Y_va_hat-Y_va).T,Y_va_hat-Y_va)/(2*np.shape(Y_va)[0]) + alpha*(np.dot(w.T,w)/(2*np.shape(Y_va)[0]))

                # Updating Cost and Hyperparameters
                if(cost<mincost):
                    mincost=cost
                    H_star= [epoch, alpha, eps, mb]
                                     
print(" Grid Search Completed")
print(" Results after performing Grid Search:")
print(" Best Hyperparameters:") 
print("   Epochs= ",H_star[0])
print("   Alpha= ",H_star[1])
print("   Learning Rate= ",H_star[2])
print("   Mini Batch Size= ", H_star[3])
print(" Cost on Validation Set with Best Hyperparameters= ", mincost)
print("\n----------------------------------------------------------------------")
print("Training on Training + Validation Dataset:")
print("----------------------------------------------------------------------")

X_data= X_data.T

# Initializing Weights and Bias
w=W
b=B

# Assigning the Best Hyperparameters
epoch=H_star[0]
alpha=H_star[1]
eps=H_star[2]
mb=H_star[3]

# Number of Training Examples
N_full=np.shape(X_data)[1]

# Stochastic Gradient Descent
for i in range(epoch):
    for j in range(int(N_full/mb)):
        X_mb=X_data[:,j*mb:(j+1)*mb]
        Y_mb=Y_data[j*mb:(j+1)*mb]

        # Calculating the Gradients
        dw= (np.dot(X_mb,(np.dot(X_mb.T,w)+b-Y_mb))+alpha*w)/mb
        db= np.sum(np.dot(X_mb.T,w)+b-Y_mb)/mb

        # Updating the Weights and Bias
        w-=eps*dw
        b-=eps*db

print(" Training Completed")
print("\n----------------------------------------------------------------------")
print("Performance Evaluation")
print("----------------------------------------------------------------------")

# Calculating Unregularized MSE Loss on Validation Set
Y_te_hat= np.dot(X_te.T,w)+b
cost= np.dot((Y_te_hat-Y_te).T,Y_te_hat-Y_te)/(2*np.shape(Y_te)[0])
print(" Cost on Test Dataset= ", cost)
print("\n")      
         