import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))
fig.suptitle('Avg Matches vs Lowe\'s Ratio on LIVE and BASE model)', fontsize=10)

ax = fig.add_subplot(111)

ax.set_ylabel('No. avg matches for all images', fontsize=9)
ax.set_xlabel('Lowe\'s Ratio Values', fontsize=9)

x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y = [0, 9.9, 86.3, 165.87, 246.7, 1277.40]
plt.plot(x,y, '-ro', label="LIVE Model")
plt.axis([0, 1.1, 0, 1400])

# y_val_offset = 50000
# for xy in zip(x,y):
#     x_val = xy[0]
#     y_val = xy[1]
#     ax.annotate('%s' % y_val, xy = (x_val,y_val+y_val_offset), textcoords='data')

x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y = [0, 12.09, 86.43, 153.53, 230.36, 1279.01]
plt.plot(x,y, '-bv', label="Base Model")
plt.axis([0, 1.1, 0, 1400])

x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y = [0, 0, 0, 0, 1.03, 733.67]
plt.plot(x,y, '-gs', label="LIVE model weighted descs")
plt.axis([0, 1.1, 0, 1400])

ax.legend()

plt.show()