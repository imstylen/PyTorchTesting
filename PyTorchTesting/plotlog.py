import matplotlib.pyplot as plt
import numpy as np

logfile = open("log.log",'r')

lines = logfile.readlines();
lines = [line.strip() for line in lines]

r = np.array(lines, dtype='float')

plt.plot(r)
plt.show()


