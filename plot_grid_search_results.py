from scipy import signal
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pylab as pl

ff = lambda x, n: signal.filtfilt(np.ones(n), float(n), x)


x=[]
y=[]
hs = []

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if r'pckl' in name and r'hidden' in name and 'episodes' in name:
            print(name)
            print(name.split('hidden')[1].split('_')[0])
            hidSize = int(name.split('hidden')[1].split('_')[0])
            with open(name, 'rb') as fid:
                   dictRes = pickle.load(fid)

            scrs = ff(dictRes['all_scores'], 20)
            x.append(list(range(len(scrs))))
            y.append(scrs)
            hs.append(hidSize)



x_ = np.array(x)
y_ = np.array(y)
hs_ = np.array(hs)

ind_srt = hs_.argsort()

x_ = np.array(x_[ind_srt])
y_ = np.array(y_[ind_srt])
hs_ = np.array(hs_[ind_srt])

colors = pl.cm.jet((hs_-np.min(hs_))/(np.max(hs_)-np.min(hs_)))

fig = plt.figure(2)
for xi, yi, clr in zip(x_, y_, colors):
    plt.plot(xi, yi, color=clr)

leg = list(map(str, hs_))
plt.legend(leg)
plt.title('hidden layer size affect on learning progress')
plt.xlabel('#episode')
plt.ylabel('average score')
plt.xlim([0, 300])
plt.show()




# print('plotting learning progression VS episode of saved session (results.pckl) ...')
#
#         with open('results.pckl', 'rb') as fid:
#             dictRes = pickle.load(fid)
#         ff = lambda x, n: signal.filtfilt(np.ones(n), float(n), x)
#         plt.figure(2)
#         plt.plot(dictRes['all_scores'])
#         plt.plot(ff(dictRes['all_scores'], 20))
#         plt.legend(('epi score', 'avg 20 epi'))
#         plt.savefig('learning_rates_filtfilt20.png')
#         plt.ion()
#         plt.show()
#         plt.pause(0.001)
#         time.sleep(5)