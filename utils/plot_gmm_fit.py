#!/usr/bin/env python3
import sys

import numpy as np
import pylab as plt

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['font.size'] = 8.

import joblib

l = joblib.load(sys.argv[1])
s = np.array(l['sizes'])
print(s)
s[s > 256**2 - 1] = 256**2

nband = l['sed'].shape[2]
n = nband * 2 + 3
nn = int(np.ceil(np.sqrt(n)))
fig, ax = plt.subplots(nn, nn)
ax = ax.flatten()
ax[0].plot(l['z_base'],  l['l_values'] / s)
ax[0].set_ylabel('L')

ax[1].plot(l['z_base'],  l['iterations'])
ax[1].set_ylabel('iter')

ax[2].plot(l['z'], l['weights'])

for i in range(l['sed'].shape[2] - 1):
    ax[i + 3].plot(l['z'], l['sed'][:, :, i] - l['sed'][:, :, i + 1])
    ax[i + 3].set_xlabel('SED %s' % i)
for i in range(l['sed'].shape[2]):
    ax[nband + 3 + i].plot(l['z'], l['err'][:, :, i])
    ax[nband + 3 + i].set_xlabel('Err %s' % i)
plt.tight_layout()
if len(sys.argv) > 2:
    plt.savefig(sys.argv[2])
else:
    plt.show()
