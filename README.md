============================================================================

Software Development for Conceptual Design and Optimization of Jet Transport Aircraft

============================================================================

It is structured into three phases that allow for design exploration within a single, consistent framework.
The software conducts aircraft performance and components mass calculation.

User have to run the GUI_Phase1.py only, to activate the software GUI
Once the python file was activated, a GUI will pops up
in the GUI users can adjust all the variables, that adjust the size and other parameters of the aircraft
then on the user can proceed to Phase 2 and Phase 3 without activating the other files

If the user chooses to activate GUI_Phase2.py, the user will be able to only access the 2nd phase
of optimization and 3rd phase.

1st Phase of optimization
The software will try to explore the according to the bound, equality, and inequality constraints
to explore and find the optimal design point. it will start exploring from the initial guess
given by the user.
The initial guess has to be edited directly in the InputParameters.py file along with other parameters

2nd Phase of optimization
the software will iterate the cruise Mach number to explore different performance and optimal design point

3rd Phase of optimization
the software will iterate both cruie Mach number and cruise altitude to explore the optimal design point 
in different cruise condition and plot a carpet plot

The carpet plot will pinpoint exactly the most optimum design point (global optimum point) 
and the realistic optimum design point (design point).

