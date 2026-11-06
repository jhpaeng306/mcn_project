# V1 cell-type and connectivity evidence

`billeh_17_cell_classes.csv` is the complete layer-by-cardinal-class inventory
used by Billeh et al. (Neuron 2020).  It is the finest V1 partition for which a
complete, runnable circuit model and recurrent graph are publicly available.

`billeh_connection_probabilities_75um.csv` transcribes Fig. 4A of that paper.
Rows are presynaptic sources and columns are postsynaptic targets.  Values are
connection probabilities at 75 micrometres.  A blank is an undetermined entry,
not a measured zero; Billeh et al. set 89 of 289 undetermined entries to zero
when instantiating their model.  Explicit `0` values are retained as printed.
The probabilities are not Wilson-Cowan gains: synaptic strength, population
density, convergence, response gain, delays, and short-term dynamics are still
required before they become a neural-field weight.

The Allen VISp/ALM SMART-seq taxonomy contains 133 transcriptomic types: 56
glutamatergic, 61 GABAergic, and 16 non-neuronal types.  No experimentally
complete 133-by-133 physiological connection matrix exists.  MICrONS resolves
more than 0.5 billion synapses in one mouse visual-cortex volume and provides
cell-level connectivity, but the mapping from every reconstructed cell to each
fine transcriptomic type and a population-level dynamical gain remains
incomplete.  The project therefore applies the amplitude calculation at a
five-population functional resolution (E, PV, SST-GER, SST-local, VIP) while
retaining the 17-class matrix as an auditable anatomical constraint.

Primary sources:

- Billeh et al., Neuron 106, 388-403 (2020), DOI 10.1016/j.neuron.2020.01.040.
- Tasic et al., Nature 563, 72-78 (2018), DOI 10.1038/s41586-018-0654-5.
- MICrONS Consortium, Nature 640, 435-447 (2025), DOI 10.1038/s41586-025-08790-w.
- Schneider-Mizell et al., Nature 640, 448-458 (2025), DOI 10.1038/s41586-024-07780-8.
- Drinnenberg et al., bioRxiv (2025), DOI 10.1101/2025.10.21.683734.
