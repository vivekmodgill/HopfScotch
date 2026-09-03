# HopfScotch
Code accompanying:
**How stimulation waveform shape affects collective oscillations in the brain networks**
Vivek Sharma, Paul H. E. Tiesinga, Joana Cabral
Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, Netherlands; Department of Bioengineering and Institute of Systems and Robotics, LARSyS, Instituto Superior Tecnico, University of Lisbon, Lisbon, Portugal
Preprint: [bioRxiv 2026.06.16.732561](https://www.biorxiv.org/content/10.64898/2026.06.16.732561v1)
Status: submitted, currently under review.
## Overview
This repository contains the code used to model how the shape of transcranial alternating current stimulation (tACS) waveforms affects whole-brain oscillatory dynamics. We use a network of delay-coupled Stuart-Landau oscillators, operating near a Hopf bifurcation and constrained by empirical human structural connectivity, fitted to resting-state MEG functional connectivity (Human Connectome Project). Five stimulation waveforms: sinusoidal, square, triangular, pulsed, and sawtooth, matched in frequency and peak amplitude, are compared for their effects on global synchrony, metastability, spectral entropy, and metastable oscillatory modes.
## Data
This project uses:
- **MEG data and structural connectivity**: [Human Connectome Project](https://www.humanconnectome.org/) (source-reconstructed resting-state MEG and diffusion-derived structural connectivity, as provided by Castaldo et al., 2023, *NeuroImage*; access requires a data-sharing agreement with HCP, not redistributed here)
Raw data are not included in this repository due to data-sharing restrictions. See the paper's Methods section for full acquisition and access details.

## Citation
If you use this code, please cite the preprint:
```bibtex
@article{sharma2026waveform,
  author  = {Sharma, Vivek and Tiesinga, Paul H. E. and Cabral, Joana},
  title   = {How stimulation waveform shape affects collective oscillations in the brain networks},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.06.16.732561}
}
```
(Citation will be updated with the final published reference upon acceptance.)
## Contact
Vivek Sharma vivek.sharma@donders.ru.nl
Joana Cabral joanabcabral@tecnico.ulisboa.pt \
Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, Netherlands \
Department of Bioengineering and Institute of Systems and Robotics, LARSyS – Laboratory for Robotics and Engineering Systems, Instituto Superior Tecnico, University of Lisbon, Lisbon, Portugal
## License
MIT License: see [LICENSE](LICENSE) for details.
