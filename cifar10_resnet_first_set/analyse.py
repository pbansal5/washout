import numpy as np

adv_egs = np.load('stats/adv_egs.npy')
array_eps = np.load('stats/adv_eps.npy')
array = np.load('stats/num_forget.npy')

for i in range(0,1300,100):
    temp = array_eps[i:i+100]
    print (int(i/100),"{:.4f}".format(np.mean(temp)),"{:.4f}".format(np.std(temp)))
exit()
# array = np.load('stats/num_forget.npy')
# final_examples = []
# for x in range(13):
#     final_examples.append(np.where(array==x)[0][:100])
# final_examples = np.concatenate(final_examples)
print (np.min(array))
print (np.max(array))
print (array.shape)
print (np.histogram(array,bins = int(np.max(array)-np.min(array))))
#print (np.where(array<=1)[0])
