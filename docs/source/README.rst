
==========================================================
railFE: Simplified Vehicle Track Interaction Model
==========================================================
Dynamic simulation of a simplified rail vehicle rolling on a railway track. The track is modelled as a timoshenko rail beam on discrete sleeper supports. The rail vehicle is modelled as a quarter car model.

==========================================================
Table of Contents
==========================================================


- :ref:`Background <introduction:background>`
- :ref:`Install <introduction:install>`
- :ref:`Usage <introduction:usage>`
	- :ref:`Examples <introduction:examples>`
- :ref:`Maintainers <introduction:maintainers>`
- :ref:`Contributing <introduction:contributing>`
- :ref:`License <introduction:license>`

==========================================================
Background
==========================================================

.. image:: ../../figs/VehicleTrackFEModel.png 
	:target: Vehicle-Track FE Model

This model can be used to simulate the high frequency dynamics of the vertical vehicle-rail interaction.
railFE can be used to construct the state space matrices of a discretely supported railway track, of railway vehicles and solve the non-linear Herzian contact mechanics between the rail and the wheel. 
railFE can be used to extract the frequency response function at selected degrees of freedom. It further allows the solution of a linear system of coupled differential equations.

==========================================================
Model Description
==========================================================

The model is composed of several substructure components:

- Vehicle assembly or properties (i.e. from Simpack),
- Track assembly (2D FE Model of Beam: 4DOF Timoshenko elements),
- Non-linear Hertzian contact spring:

.. math:: 

		f_c = \left\{\begin{matrix}
		K_H\delta^{1.5},\; \delta>0 \\ 
		0,\; \delta\leq0 \\
		\end{matrix}\right\}.

==========================================================
System Equations and State Space representation:
==========================================================

The equilibrium matrices of the system are formulated as: 

.. math:: 

		M_{sys}\ddot{q}_{sys}+C_{sys}\dot{q}_{sys}+[K_{sys}-K_c\delta^{0.5}E]q_{sys}=f_{irr}+f_{ext}


Local dynamics, modal superposition:

.. math:: 

		\ddot{\eta}_{i}+2\zeta_{i}\omega_{i}\dot{\eta}_{i}+\omega_{i}^2\eta_{i}+M_{cross}\ddot{q}_{tr}=f_{i}


==========================================================
Install
==========================================================

The following steps provide guidance on how to install railFE:

1. Install Python, required Python packages, and get the railFE source code from GitHub
2. Install railFE

Once you have installed the aforementioned tools follow these steps to build and install railFE:

* Open a Terminal (Linux/macOS) or Command Prompt (Windows), navigate into the top-level railFE directory and activate your environment of choice. Run the following command:

.. code-block:: bash

	(railFE)$ python setup.py install

**You are now ready to proceed to running railFE.**

==========================================================
Usage
==========================================================

railFE is designed as an extensible Python package. 
To use railFE, see the examples and python api.

----------------------------------------------------------
Examples
----------------------------------------------------------

The folder railFE/examples contains several usage cases of the package:

1. :ref:`Example 1 <shapefunctions>`: Plotting the shape function for 4DOF Timoshenko elements without and with elastic bedding.
2. :ref:`Example 2 <track_freqresponse>`: Evaluation of the frequency response of the track (selected observed degrees of freedom) under a point load applied at a fixed location on the Finite Element model.  
3. :ref:`Example 3 to do <examples:to_do>`: Simulation of dynamic response of the system with gaussian track noise. 
4. :ref:`Example 4 <timeintegration_impulse>`: Simulation of dynamic response when crossing a geometric irregularity on the rail (impulse like excitation and gaussian noise).
5. :ref:`Example 5 <timeintegration_varying>`: Simulation of dynamic response when crossing a geometric irregularity on the rail with different track parameters and speeds  (impulse like excitation and gaussian noise).
6. :ref:`Example 6 <TimoshenkoBeam_AnalyticShapeFunctions.py>`: Analytic solution of the timoshenko beam shape functions: TimoshenkoBeam_AnalyticShapeFunctions.py

==========================================================
Maintainers
==========================================================

`@CyprienHoelzl <https://github.com/CyprienHoelzl/>`_

==========================================================
Contributing
==========================================================

Feel free to dive in! `Open an issue <https://github.com/CyprienHoelzl/railFE/issues/new>`_ or submit PRs.

----------------------------------------------------------
Contributors
----------------------------------------------------------

This project exists thanks to all the people who contribute.

==========================================================
License
==========================================================

`MIT <../../LICENSE>`_ © Cyprien Hoelzl