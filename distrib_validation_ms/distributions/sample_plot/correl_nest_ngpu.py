import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

y1mean = np.zeros((254, 400))
y2mean = np.zeros((254, 400))
y3mean = np.zeros((254, 400))
y1std = np.zeros((254, 400))   
y2std = np.zeros((254, 400))   
y3std = np.zeros((254, 400))                 

path='../../../data/'

for i in range(8):
    y1vals = np.zeros((400,10))
    y2vals = np.zeros((400,10))
    y3vals = np.zeros((400,10))
    for i_run in range(10):
        fn1 = path+'mam_original_ms3/data' + str(i_run) + '/correl2_' + str(i) \
                + '.dat'
        fn2 = path+'mam_ngpu_ms/data' + str(i_run) + '/correl_ngpu_' + \
        str(i) + '.dat'
        fn3 = path+'mam_original_ms4/data' + str(i_run) + '/correl1_' + str(i) \
                + '.dat'

        data1 = np.loadtxt(fn1)
        data2 = np.loadtxt(fn2)
        data3 = np.loadtxt(fn3)

        x1=[row[0] for row in data1]
        y1=[row[1] for row in data1]

        x2=[row[0] for row in data2]
        y2=[row[1] for row in data2]

        x3=[row[0] for row in data3]
        y3=[row[1] for row in data3]

        f1 = interpolate.interp1d(x1, y1,fill_value="extrapolate")
        f2 = interpolate.interp1d(x2, y2,fill_value="extrapolate")
        f3 = interpolate.interp1d(x3, y3,fill_value="extrapolate")

        xnew = np.linspace(-0.01, 0.15, 400)

        y1new = f1(xnew)
        y2new = f2(xnew)
        y3new = f3(xnew)

        eps = 1.0e-3
        for j in range(len(y2new)):
            y3new[j] = max(y3new[j], eps)
            y2new[j] = max(y2new[j], eps)
            y1new[j] = max(y1new[j], eps)
            y1vals[j][i_run] = y1new[j]
            y2vals[j][i_run] = y2new[j]
            y3vals[j][i_run] = y3new[j]
            y1mean[i][j] = y1mean[i][j] + y1new[j]/10 
            y2mean[i][j] = y2mean[i][j] + y2new[j]/10
            y3mean[i][j] = y3mean[i][j] + y3new[j]/10
    for j in range(300):
        y1std[i][j] = np.std(y1vals[j][:])
        y2std[i][j] = np.std(y2vals[j][:])
        y3std[i][j] = np.std(y3vals[j][:])

ticks=15
legenda=15
label=14
poplabel=["L2/3E", "L2/3I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
popname=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]

for i in range(8):
    plt.figure(i+1)
    plt.plot([0], [0], marker=' ', linestyle=' ', label=poplabel[i])
    plt.plot(xnew, y1mean[i], color="navy", label="NEST 1")
    plt.fill_between(xnew, y1mean[i]-y1std[i], y1mean[i]+y1std[i], color="navy", alpha = 0.25)
    plt.plot(xnew, y3mean[i], color="cornflowerblue", label="NEST 2")
    plt.fill_between(xnew, y3mean[i]-y3std[i], y3mean[i]+y3std[i], color="cornflowerblue", alpha = 0.25)
    plt.plot(xnew, y2mean[i], '--', color="red", label="NEST GPU")
    plt.fill_between(xnew, y2mean[i]-y2std[i], y2mean[i]+y2std[i], color="red", alpha = 0.25)
    plt.xlabel("correlation", fontsize=label)
    plt.tick_params(labelsize=ticks)
    plt.legend(fontsize=legenda)
    plt.grid()
    plt.savefig("plot_dist/correlation/band_"+popname[i]+"2NEST", dpi=300)

plt.draw()
plt.pause(0.5)
input("Press Enter to continue...")
plt.close()


