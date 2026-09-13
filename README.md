# jsn-sim
a simulation to test combinations of four jovian symbolic navigators, to estimate how well they can move around in three-dimensions.

The simulation creates a sphere of 300 vectors as targets. The simulation then tests every possible combination of 4 navigators,
with up to 5 activations, to see which target vectors can be achieved within a 20 degree margin of error. Note that there are 
variables to change the number of targets, angle limit, and the number of activations. Have not tested different values of these, 
but they should work if you want to change something.

The simulation also checks if the vector achieved exceeds the activation limit of the navigators. The final activation is 
allowed to exceed this limit, but intermediate activations are not. The activation limit is also controlled by a variable.

Default behavior is to display a number of top results, defined by the TOP_RESULTS variable. You can also specify one or more 
combinations of four navigators, to see how they rank.

Upon finishing the simulation, results are saved in a file as defined by the RESULTS_FILE variable. Subsequent runs will check for 
this results file and will reuse those results if the file is located.
