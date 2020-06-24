#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Loading training data from file
#----------------------------------------------------------------
trn=np.genfromtxt('trainX.txt',delimiter=' ')

#----------------------------------------------------------------
# Calculating probabilities for class: 2
#----------------------------------------------------------------
two_trn=trn[:250,:]
two_one_prob=two_trn.sum(axis=0)/two_trn.shape[0]
two_zero_prob=1-two_one_prob

#----------------------------------------------------------------
# Calculating probabilities for class: 4
#----------------------------------------------------------------
four_trn=trn[250:,:]
four_one_prob=four_trn.sum(axis=0)/four_trn.shape[0]
four_zero_prob=1-four_one_prob

#----------------------------------------------------------------
# Loading testing data from file
#----------------------------------------------------------------
tst_x=np.genfromtxt('testX.txt',delimiter=' ')
tst_y=np.genfromtxt('testY.txt',delimiter=' ')

#----------------------------------------------------------------
# Performing calculations on testing data
#----------------------------------------------------------------
index=0
tp=tn=fp=fn=0
prdct=None
for (tstx,actual) in zip(tst_x,tst_y):
    two_prdct=0.5
    four_prdct=0.5
    one_index=np.where(tstx==1)
    zero_index=np.where(tstx==0)
    two_1=two_one_prob[one_index]
    two_0=two_zero_prob[zero_index]
    four_1=four_one_prob[one_index]
    four_0=four_zero_prob[zero_index]
    two_prdct=two_prdct*np.prod(two_1)*np.prod(two_0)
    four_prdct=four_prdct*np.prod(four_1)*np.prod(four_0)
    
    if two_prdct>four_prdct:
        print("Predicted Class: 2           Actual Class:",actual.astype(int))
        if (actual==2):
            tp=tp+1
        else:
            fp=fp+1
    else:
        print("Predicted Class: 4           Actual Class:",actual.astype(int))
        if (actual==4):
            tn=tn+1
        else:
            fn=fn+1
    
    plt.imshow(np.reshape(tstx,(16,16),order='F'))
    plt.show()
    print('---------------------------------------------',end='\n\n')
    
#----------------------------------------------------------------
# Computing accuracy of model
#----------------------------------------------------------------            
accuracy=(tp+tn)/(tp+tn+fp+fn)
print(' _________________',end='\n\n')
print('| ',end='')
print('Accuracy: ',accuracy,end='')
print('  |')
print(' _________________')

