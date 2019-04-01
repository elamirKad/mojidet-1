import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import csv

with open("database.csv", "r") as f1:
    reader = csv.reader(f1)
    next(reader, None)
    neg = [[],[],[]]
    for a, b, c in reader:
        neg[0].append(float(a))
        neg[1].append(float(b))
        neg[2].append(float(c))
    plt.plot(neg[0],'r-', label = 'negative')
    plt.plot(neg[1],'gray', label = 'neutral')
    plt.plot(neg[2],'g-', label = 'positive')
    plt.legend()
    plt.show()
        