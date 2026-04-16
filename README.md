============================================================================

Software Development for Conceptual Design and Optimization of Jet Transport Aircraft

============================================================================

It is structured into three phases that allow for design exploration within a single, consistent framework.
The software conducts aircraft performance and components mass calculation.

User have to run the GUI_Phase1.py only, to activate the software GUI
Once the python file was activated, a GUI will pops up
where users can give input as an initial guess of  the variables,
then on the user can proceed to Phase 2 and Phase 3 without activating the other files

If the user chooses to activate GUI_Phase2.py, the user will be able to only access the 2nd phase
of optimization and 3rd phase.

1st Phase of optimization
The software will try to explore the design space according to the bound, equality, and inequality constraints
to explore and find the optimal design point. It will start exploring from the initial guess
given by the user.

2nd Phase of optimization
The software will vary the cruise Mach number within the defined range to explore different performance and optimal design configuration.

3rd Phase of optimization
The software will vary both cruise Mach number and cruise altitude to explore the optimal design point in different cruise conditions and plot a carpet plot

The carpet plot will pinpoint exactly the optimum design point (global optimum point) 
and the realistic optimum design point (design point).

