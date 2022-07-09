# railFE: Simplified Vehicle Track Interaction Model
Dynamic simulation of a simplified rail vehicle rolling on a railway track. The track is modelled as a timoshenko beam rail on discrete sleeper supports. The rail vehicle is modelled as a quarter car model.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Examples](#examples)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

![Vehicle-Track FE Model](figs/VehicleTrackFEModel.png)
This model can be used to simulate the high frequency dynamics of the vertical vehicle-rail interaction.

## Model Description

The model is composed of several substructure components:
- Vehicle assembly or properties (i.e. from Simpack),
- Track assembly (2D FE Model of Beam: 4DOF Timoshenko elements),
- Non-linear Hertzian contact spring:  

```math
f_c = \left\{\begin{matrix}K_H\delta^{1.5},\; \delta>0 \\ 0,\; \delta\leq0 \\\end{matrix}\right.
```

## System Equations and State Space representation:
The equilibrium matrices of the system are formulated as: 

```math
M_{sys}\ddot{q}_{sys}+C_{sys}\dot{q}_{sys}+[K_{sys}-K_c\delta^{0.5}E]q_{sys}=f_{irr}+f_{ext}
```

Local dynamics, modal superposition:
```math
\ddot{\eta}_{i}+2\zeta_{i}\omega_{i}\dot{\eta}_{i}+\omega_{i}^2\eta_{i}+M_{cross}\ddot{q}_{tr}=f_{i}
```

## Install

The following steps provide guidance on how to install railFE:

1. Install Python, required Python packages, and get the railFE source code from GitHub
2. Install railFE

Once you have installed the aforementioned tools follow these steps to build and install railFE:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level railFE directory and activate your environment of choice. Run the following command:
```
(railFE)$ python setup.py install
```

**You are now ready to proceed to running railFE.**

## Usage
railFE is designed as an extensible Python package.

### Examples

The folder railFE/examples contains several usage examples of the 
1. [Example 1](examples/TrackFrequencyResponseEvaluation.py): Evaluation of the frequency response of the track (selected observed degrees of freedom) under a point load applied at a fixed location on the Finite Element model.  
2. [Example 2 to do](examples/to_do): Simulation of dynamic response of the system with gaussian track noise. 
3. [Example 3](examples/AnalysisSimplifiedVTIM.py): Simulation of dynamic response when crossing a geometric irregularity on the rail (impulse like excitation).
4. [Example 4](examples/TimoshenkoBeam_AnalyticShapeFunctions.py) Analytic solution of the timoshenko beam shape functions: TimoshenkoBeam_AnalyticShapeFunctions.py

## Maintainers

[@CyprienHoelzl](https://github.com/CyprienHoelzl/).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/CyprienHoelzl/railFE/issues/new) or submit PRs.

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/CyprienHoelzl/railFE/graphs/contributors"><img src="https://opencollective.com/railFE/contributors.svg?width=890&button=false" /></a>

## License

[MIT](LICENSE) © Cyprien Hoelzl