import numpy as np

array = np.load('stats/num_forget.npy')
print (np.min(array))
print (np.max(array))
print (array.shape)
#print (np.histogram(array,bins = int(np.max(array)-np.min(array))))
print (np.where(array<=1)[0])
