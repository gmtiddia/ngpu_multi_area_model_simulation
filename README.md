# NEST GPU implementation of the Multi-Area Model

This is the NEST GPU implementation of the Multi-Area Model of the macaque visual cortex. The model has been developed at the Institute of Neuroscience and Medicine (INM-6), Research Center Jülich and [here](https://github.com/INM-6/multi-area-model) you can find the original implementation for the CPU version of the NEST simulator.

The NEST GPU implementation has been reported in the following publication:

- Tiddia, G., Golosio, B., Albers, J., Senk, J., Simula, F., Pronold, J., Fanti, V., Pastorelli, E., Paolucci, P. S., & Van Albada, S. J. (2022). Fast Simulation of a Multi-Area Spiking Network Model of Macaque Cortex on an MPI-GPU Cluster. Frontiers in Neuroinformatics, 16, 883333. https://doi.org/10.3389/fninf.2022.883333

The code employed to obtain the result in the publication above can be found in [this release](https://github.com/gmtiddia/ngpu_multi_area_model_simulation/releases/tag/v_Tiddia2022), whereas the code in this repository enables the simulation of the model implemented in NEST GPU.

To analyze the distribution of the spiking activity and obtaining a validation as the one shown in the publciation above please use the code contained in [this repository](https://github.com/gmtiddia/ngpu_mam_validation).



## Requirements

Among the requirements we have

- Python 3
- python_dicthash (https://github.com/INM-6/python-dicthash)
- correlation_toolbox (https://github.com/INM-6/correlation-toolbox)
- pandas
- numpy
- nested_dict
- matplotlib (2.1.2)
- scipy 
- pytest

To install the requirement packages with pip, execute

``pip install -r requirements.txt``

Please note that the NEST GPU simulator must be installed separately, see [the installation instructions](https://nest-gpu.readthedocs.io/en/latest/installation/index.html).


## Content

















## Contributors

All the authors of the publication made contributions to the scientific content. The code was written by Bruno Golosio and Gianmarco Tiddia.


## Contact

Gianmarco Tiddia, Istituto Nazionale di Fisica Nucleare, Sezione di Cagliari, Italy, gianmarco.tiddia@ca.infn.it

## Citation

If you use this code, please cite the paper indicated above in your publication. For the model itself, please refer to the [citation section](https://github.com/INM-6/multi-area-model/tree/master?tab=readme-ov-file#citation) of the original model.

