import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
#
# plt.style.use('ggplot')
#
# print(plt.style.available)
#
# plt.figure(figsize=(11,4), dpi=100)
#
# N = 7
# inliers_0 = (93 , 95 , 98 , 100 , 101 , 102 , 104)
# inliers_1 = (81 , 83 , 83 , 85 , 88 , 89 , 89 )
#
# ind = np.arange(0,3*N, 3)
# width = 0.8
# plt.bar(ind, inliers_0, width, label='Inliers for vanilla RANSAC')
# plt.bar(ind + width, inliers_1, width, label='Inliers for PROSAC (lowe\'s ratio by reliability score ratio)')
#
# plt.ylabel('Inliers - Live Model', fontsize=20)
# # plt.title('Inliers for vanilla RANSAC and PROSAC against live model', fontsize=20)
#
# plt.xticks(ind + width/2, ('+s1', '+s2', '+s3', '+s4', '+s5', '+s6', '+s7'))
# plt.legend(loc='upper center', framealpha=1, fontsize=16, shadow = True)
#
# ax = plt.gca()
# ax.tick_params(axis="y", labelsize=18)
# ax.tick_params(axis="x", labelsize=18)
# ax.set_ylim([60,126])
#
# plt.savefig('/Users/alex/Projects/EngDLocalProjects/Papers Local - Before iCloud/paper/figures/inliers_figure.png')
# plt.show()

# this or above ---------------------------

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

print(plt.style.available)

# This is for vanillia RANSAC - Live model
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.figure(figsize=(8,4), dpi=100)
N = 8
time = (0.41, 0.22 , 0.19 , 0.17 , 0.12 , 0.14 , 0.14 , 0.12)
# trans_err = (0.07, 0.04 , 0.03 , 0.03 , 0.03 , 0.02 , 0.02 , 0.02)
# rot_err = np.array([5.01, 3.61 , 3.62 , 3.29 , 2.75 , 2.43 , 2.41 , 2.18]) / 20

# This is for PROSAC version that uses the inverse lowe's ratio by the reliability score ratio - Live model
# plt.figure(figsize=(11,4), dpi=100)
# N = 7
# time = (0.09 , 0.09 , 0.09 , 0.05 , 0.07 , 0.06 , 0.05)
# trans_err = (0.05 , 0.05 , 0.05 , 0.04 , 0.04 , 0.04 , 0.04)
# rot_err = np.array([4.68 , 4.64 , 4.87 , 4.13 , 4.06 , 3.92 , 3.79]) / 60

ind = np.arange(N)
width = 0.8
plt.bar(ind, time, width, label='Time', color='firebrick')
# plt.bar(ind + width, trans_err, width, label='Translation Error')
# plt.bar(ind + 2 * width, rot_err, width, label='Rotation Error')

plt.ylabel('Seconds', fontsize=20)
# plt.title('Values for vanilla RANSAC against live model', fontsize=20)

plt.xticks(ind, ('Base', '+s1', '+s2', '+s3', '+s4', '+s5', '+s6', '+s7'))

plt.legend(loc='best', framealpha=1, fontsize=16, shadow = True)

ax = plt.gca()
ax.tick_params(axis="y", labelsize=18)
ax.tick_params(axis="x", labelsize=18)

plt.savefig('/Users/alex/Projects/EngDLocalProjects/Papers Local - Before iCloud/paper/figures/inliers_figure.png')
plt.show()

# ---------------------------

