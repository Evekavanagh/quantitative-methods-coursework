import math as m
import numpy as np
from scipy.stats import norm,pearsonr
import matplotlib.pyplot as plt

S = 60          
K = 50
H1 = 65
H2 = 75
r = 0.05
T = 1
sigma = 0.3
n = 260
nr=5000
dt = T/n
# create condition counters
C1 = np.zeros((nr,1))
C2 = np.zeros((nr,1))

# create result matrices
S_val = np.zeros((nr,n+1))
FinPayOff = np.zeros((nr,1))
AvgFinPayOff = np.zeros((int(nr/2),1))
rand = np.random.randn(int(nr/2),n+1)
control = np.zeros((nr,1))
AvgControl = np.zeros((int(nr/2),1))


#calculate share price at each time step
S_val[:,0]=S
for i in range(0,nr,2):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1]*m.exp((r-0.5*sigma**2)*dt+sigma*rand[int(i/2),j-1]*m.sqrt(dt)) 
        S_val[i+1,j] = S_val[i+1,j-1]*m.exp((r-0.5*sigma**2)*dt+sigma*-rand[int(i/2),j-1]*m.sqrt(dt))   

for i in range(0,nr):
    for j in range(n+1):
        if S_val[i,j]>H2:
          C2[i,0] +=1
          C1[i,0] +=1 
        elif S_val[i,j]>H1:
          C1[i,0] +=1 
        
          
count = nr
for i in range(0,nr):
    if S_val[i,n]>K:
        control[i,0] = S_val[i,-1] - K 
        if C2[i,0] >=150:
            FinPayOff[i,0] = S_val[i,n] - K +30
        elif C1[i,0] >=100 and C2[i,0]<150 and ((C1[i,0]+C2[i,0])/2)<125:
            FinPayOff[i,0] = S_val[i,n] - K +10
        elif C1[i,0] >=100 and C2[i,0] < 150:
            FinPayOff[i,0]  =S_val[i,n] -K
        elif C1[i,0] < 100:
            FinPayOff[i,0] = 10
    else:
        FinPayOff[i,0] = 0
        count += -1
        
for i in range(0,nr,2):   
    AvgFinPayOff[int(i/2),0] = 0.5*(FinPayOff[i,0]+FinPayOff[i+1,0])
    AvgControl[int(i/2),0] = 0.5*(control[i,0]+control[i+1,0])   
    
PDisc = np.exp(-r*T)*AvgFinPayOff
Price = np.mean(PDisc)
std1 = np.std(PDisc)
        
print('The price of the option before control variate is %.5f.'%Price)
print('The std of the option before control variate is %.5f.'%std1)
print('The option is exercised',count,'times i.e.',(count*100)/nr,'% of the time.')

#implement control variate technique

#calculate Black Scholes Call Price, which will act as the expected value of our control variate, the vanilla call price
q = 0
d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
BSprice = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


Discontrol = np.exp(-r*T)*AvgControl #discount the control payoff
b = np.cov(PDisc.flatten(),Discontrol.flatten())[0,1]/np.var(Discontrol.flatten()) #calculate b
PriceControl = PDisc -b*(Discontrol-BSprice) #implement the control variate technique to calculate option price for each simulation
mean,std = norm.fit(PriceControl)
corr,_ = pearsonr(PDisc.flatten(),Discontrol.flatten())

print('The correlation between the option price and the control variate is %5f.'%corr)
print('The price of the option after variance reduction is %.5f'%mean)#,'with std %.5f.'%std)
print('The std of the option before control variate is %.5f.'%std)

plt.figure()
for i in range(nr):
    plt.plot(S_val[i,:])
