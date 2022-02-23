# Data analysis

Subfolders description:
- ``distributions``: the files firing_rate.py, cv_isi.py and correl.py compute the smoothed histograms of respectively firing rate, CV ISI and Pearson correlation from the simulations spike recordings
- ``earth-mover-distance``: the files emd_firing_rate.py, emd_cv_isi.py and emd_corr.py compute the Earth Mover's Distance (EMD) between the distributions of the same model population that belong to different simulations
  - ``raw_dist``: the Python scripts compute the values of firing rate, CV ISI and Pearson correlation for each population of the model without storing them in histograms
    -  ``violin_plot_dataset`` there are the scripts that take in input the values of the above distributions and stores them in csv files to compute the distributions violin plots
- ``dist_plots``: 
  - ``distrib``: the scripts dist_firing_rate.py, dist_cv_isi.py and dist_corr.py take in input the smoothet histograms obtained with the files in the ``distributions`` folder and produce the plots, here presented in ``Areas`` for both the ground and the metastable states (suffix gs and ms respectively)
    -  ``sample_plot`` stores the scripts that reproduce the sample plots of the distributions (Fig. 2 of the manuscript)
  - ``violinplot``: the script violinplot.py takes in input the violin plot data described above and produce the plots violinplot_gs.png and violinplot_ms.png
  - ``emd/boxplots``: the script boxplots.py computes the EMD boxplots. The folder stores both the data used to reproduce the boxplots and the figures
- ``eval_time``: the subfolders ``ground_state`` and ``metastable_state`` are structured in a totally similar way. The files get_times_NEST.py and get_times_NGPU.py extract the Building time and the Simulation time of the simulations and store in the csv files the average and the standard deviation for every time in exam

