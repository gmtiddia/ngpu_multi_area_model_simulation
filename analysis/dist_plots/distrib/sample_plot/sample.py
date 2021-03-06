import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import sys

def extract_dist(n1, n2, n3, len_arr, xmin, xmax):
    path=''
    y1mean = np.zeros((8, len_arr))
    y2mean = np.zeros((8, len_arr))
    y3mean = np.zeros((8, len_arr))
    y1std = np.zeros((8, len_arr))
    y2std = np.zeros((8, len_arr))
    y3std = np.zeros((8, len_arr)) 
    for i in range(8):
        y1vals = np.zeros((len_arr,10))
        y2vals = np.zeros((len_arr,10))
        y3vals = np.zeros((len_arr,10))
        for i_run in range(10):
            fn1 = path + 'dist_dataset1/data' + str(i_run) + n1 + str(i) + '.dat'
            fn3 = path + 'dist_dataset2/data' + str(i_run) + n3 + str(i) + '.dat'
            fn2 = path + 'dist_dataset3/data' + str(i_run) + n2 + str(i) + '.dat'
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

            xnew = np.linspace(xmin, xmax, len_arr)

            y1new = f1(xnew)
            y2new = f2(xnew)
            y3new = f3(xnew)

            eps = 1.0e-3
            for j in range(len(y2new)):
                y2new[j] = max(y2new[j], eps)
                y1new[j] = max(y1new[j], eps)
                y3new[j] = max(y3new[j], eps)
                y1vals[j][i_run] = y1new[j]
                y2vals[j][i_run] = y2new[j]
                y3vals[j][i_run] = y3new[j]
                y1mean[i][j] = y1mean[i][j] + y1new[j]/10 
                y2mean[i][j] = y2mean[i][j] + y2new[j]/10
                y3mean[i][j] = y3mean[i][j] + y3new[j]/10
        for j in range(len_arr):
            y1std[i][j] = np.std(y1vals[j][:])
            y2std[i][j] = np.std(y2vals[j][:])
            y3std[i][j] = np.std(y3vals[j][:])

    return(xnew, y1mean, y2mean, y3mean, y1std, y2std, y3std)


fn1 = '/firing_rate_'
fn2 = '/firing_rate_'
fn3 = '/firing_rate_'
fr, y1_fr, y2_fr, y3_fr, y1_fr_std, y2_fr_std, y3_fr_std = extract_dist(fn1, fn2, fn3, 300, 0.0, 100.0)

fn1 = '/cv_isi_'
fn2 = '/cv_isi_'
fn3 = '/cv_isi_'
cv, y1_cv, y2_cv, y3_cv, y1_cv_std, y2_cv_std, y3_cv_std = extract_dist(fn1, fn2, fn3, 300, 0.0, 5.0)

fn1 = '/correl_'
fn2 = '/correl_'
fn3 = '/correl_'
co, y1_co, y2_co, y3_co, y1_co_std, y2_co_std, y3_co_std = extract_dist(fn1, fn2, fn3, 400, -0.01, 0.15)


id_L4E = 2
id_L4I = 3

ticks=25
legenda=25
label=25
poplabel=["L4E", "L4I"]
popname=["L4E", "L4I"]
fig, ax = plt.subplots(3,2)
dashstyle = [5, 4, 5, 4]
dashstyle1 = [6, 1, 6, 1]
linew = 2.75
color1 = "#72b6a1"
color2 = "mediumseagreen"
color3 = "#ea9474"

ax[0,0].set_title("Population " + poplabel[0], fontsize=label+5)
ax[0,0].set_xlabel("Firing rate [spikes/s]", fontsize=label)
ax[0,0].set_ylabel("p", fontsize=label)
ax[0,0].grid()
ax[0,0].plot(fr, y1_fr[id_L4E], color=color1, label="NEST 1", linewidth = linew)
ax[0,0].plot(fr, y3_fr[id_L4E], "--", color=color2, label="NEST 2",  linewidth = linew, dashes=dashstyle1)
ax[0,0].plot(fr, y2_fr[id_L4E], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[0,0].fill_between(fr, y1_fr[id_L4E]-y1_fr_std[id_L4E], y1_fr[id_L4E]+y1_fr_std[id_L4E], color=color1, alpha = 0.2, lw=0)
ax[0,0].fill_between(fr, y2_fr[id_L4E]-y2_fr_std[id_L4E], y2_fr[id_L4E]+y2_fr_std[id_L4E], color=color3, alpha = 0.2, lw=0)
ax[0,0].fill_between(fr, y3_fr[id_L4E]-y3_fr_std[id_L4E], y3_fr[id_L4E]+y3_fr_std[id_L4E], color=color2, alpha = 0.2, lw=0)
ax[0,0].tick_params(labelsize=ticks)
ax[0,0].text(-0.125,1.03,"A", transform=ax[0,0].transAxes, weight="bold", fontsize=label)

ax[0,1].set_title("Population " + poplabel[1], fontsize=label+5)
ax[0,1].set_xlabel("Firing rate [spikes/s]", fontsize=label)
ax[0,1].set_ylabel("p", fontsize=label)
ax[0,1].grid()
ax[0,1].plot(fr, y1_fr[id_L4I], color=color1, label="NEST 1", linewidth = linew)
ax[0,1].plot(fr, y3_fr[id_L4I], "--", color=color2, label="NEST 2", linewidth = linew, dashes=dashstyle1)
ax[0,1].plot(fr, y2_fr[id_L4I], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[0,1].fill_between(fr, y1_fr[id_L4I]-y1_fr_std[id_L4I], y1_fr[id_L4I]+y1_fr_std[id_L4I], color=color1, alpha = 0.2, lw=0)
ax[0,1].fill_between(fr, y2_fr[id_L4I]-y2_fr_std[id_L4I], y2_fr[id_L4I]+y2_fr_std[id_L4I], color=color3, alpha = 0.2, lw=0)
ax[0,1].fill_between(fr, y3_fr[id_L4I]-y3_fr_std[id_L4I], y3_fr[id_L4I]+y3_fr_std[id_L4I], color=color2, alpha = 0.2, lw=0)
ax[0,1].tick_params(labelsize=ticks)
ax[0,1].text(-0.125,1.03,"B", transform=ax[0,1].transAxes, weight="bold", fontsize=label)

#ax[1,0].set_title("CV ISI " + poplabel[0], fontsize=label)
ax[1,0].set_xlabel("CV ISI", fontsize=label)
ax[1,0].set_ylabel("p", fontsize=label)
ax[1,0].grid()
ax[1,0].plot(cv, y1_cv[id_L4E], color=color1, label="NEST 1", linewidth = linew)
ax[1,0].plot(cv, y3_cv[id_L4E], "--", color=color2, label="NEST 2", linewidth = linew, dashes=dashstyle1)
ax[1,0].plot(cv, y2_cv[id_L4E], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[1,0].fill_between(cv, y1_cv[id_L4E]-y1_cv_std[id_L4E], y1_cv[id_L4E]+y1_cv_std[id_L4E], color=color1, alpha = 0.2, lw=0)
ax[1,0].fill_between(cv, y2_cv[id_L4E]-y2_cv_std[id_L4E], y2_cv[id_L4E]+y2_cv_std[id_L4E], color=color3, alpha = 0.2, lw=0)
ax[1,0].fill_between(cv, y3_cv[id_L4E]-y3_cv_std[id_L4E], y3_cv[id_L4E]+y3_cv_std[id_L4E], color=color2, alpha = 0.2, lw=0)
ax[1,0].tick_params(labelsize=ticks)
ax[1,0].text(-0.125,1.03,"C", transform=ax[1,0].transAxes, weight="bold", fontsize=label)

#ax[1,1].set_title("CV ISI " + poplabel[1], fontsize=label)
ax[1,1].set_xlabel("CV ISI", fontsize=label)
ax[1,1].set_ylabel("p", fontsize=label)
ax[1,1].grid()
ax[1,1].plot(cv, y1_cv[id_L4I], color=color1, label="NEST 1", linewidth = linew)
ax[1,1].plot(cv, y3_cv[id_L4I], "--", color=color2, label="NEST 2", linewidth = linew, dashes=dashstyle1)
ax[1,1].plot(cv, y2_cv[id_L4I], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[1,1].fill_between(cv, y1_cv[id_L4I]-y1_cv_std[id_L4I], y1_cv[id_L4I]+y1_cv_std[id_L4I], color=color1, alpha = 0.2, lw=0)
ax[1,1].fill_between(cv, y2_cv[id_L4I]-y2_cv_std[id_L4I], y2_cv[id_L4I]+y2_cv_std[id_L4I], color=color3, alpha = 0.2, lw=0)
ax[1,1].fill_between(cv, y3_cv[id_L4I]-y3_cv_std[id_L4I], y3_cv[id_L4I]+y3_cv_std[id_L4I], color=color2, alpha = 0.2, lw=0)
ax[1,1].tick_params(labelsize=ticks)
ax[1,1].text(-0.125,1.03,"D", transform=ax[1,1].transAxes, weight="bold", fontsize=label)

#ax[2,0].set_title("Pearson correlation " + poplabel[0], fontsize=label)
ax[2,0].set_xlabel("correlation", fontsize=label)
ax[2,0].set_ylabel("p", fontsize=label)
ax[2,0].grid()
ax[2,0].plot(co, y1_co[id_L4E], color=color1, label="NEST 1", linewidth = linew)
ax[2,0].plot(co, y3_co[id_L4E], "--", color=color2, label="NEST 2", linewidth = linew, dashes=dashstyle1)
ax[2,0].plot(co, y2_co[id_L4E], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[2,0].fill_between(co, y1_co[id_L4E]-y1_co_std[id_L4E], y1_co[id_L4E]+y1_co_std[id_L4E], color=color1, alpha = 0.2, lw=0)
ax[2,0].fill_between(co, y2_co[id_L4E]-y2_co_std[id_L4E], y2_co[id_L4E]+y2_co_std[id_L4E], color=color3, alpha = 0.2, lw=0)
ax[2,0].fill_between(co, y3_co[id_L4E]-y3_co_std[id_L4E], y3_co[id_L4E]+y3_co_std[id_L4E], color=color2, alpha = 0.2, lw=0)
ax[2,0].tick_params(labelsize=ticks)
ax[2,0].text(-0.125,1.03,"E", transform=ax[2,0].transAxes, weight="bold", fontsize=label)

#ax[2,1].set_title("Pearson correlation " + poplabel[1], fontsize=label)
ax[2,1].set_xlabel("correlation", fontsize=label)
ax[2,1].set_ylabel("p", fontsize=label)
ax[2,1].grid()
ax[2,1].plot(co, y1_co[id_L4I], color=color1, label="NEST 1", linewidth = linew)
ax[2,1].plot(co, y3_co[id_L4I], "--", color=color2, label="NEST 2", linewidth = linew, dashes=dashstyle1)
ax[2,1].plot(co, y2_co[id_L4I], "--", color=color3, label="NEST GPU", dashes=dashstyle, linewidth = linew)
ax[2,1].fill_between(co, y1_co[id_L4I]-y1_co_std[id_L4I], y1_co[id_L4I]+y1_co_std[id_L4I], color=color1, alpha = 0.2, lw=0)
ax[2,1].fill_between(co, y2_co[id_L4I]-y2_co_std[id_L4I], y2_co[id_L4I]+y2_co_std[id_L4I], color=color3, alpha = 0.2, lw=0)
ax[2,1].fill_between(co, y3_co[id_L4I]-y3_co_std[id_L4I], y3_co[id_L4I]+y3_co_std[id_L4I], color=color2, alpha = 0.2, lw=0)
ax[2,1].tick_params(labelsize=ticks)
ax[2,1].text(-0.125,1.03,"F", transform=ax[2,1].transAxes, weight="bold", fontsize=label)
ax[2,1].legend(bbox_to_anchor=(-0.1, -0.3), loc='lower center', borderaxespad=0., prop={'size': legenda+2}, ncol=3)


fig.set_size_inches(23., 25.)
plt.subplots_adjust(top=0.95, wspace = 0.25, hspace = 0.25)
plt.savefig("dist_sample.png", dpi=300)

plt.show()


