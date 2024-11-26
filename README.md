# consumer-resource-plasticity

Code to recreate figures and analysis from "Resource-use plasticity governs the causal relationship between traits and community structure in model microbial communities" currently on Biorxiv.

Use test.py as an introduction to using the Community object to run plastic consumer resource models with varying numbers of species and resources. The model function in util.py is the model described in the paper.

Important functions:
model in utils.py is the main model that we derive in the supplementary text section A.
runModel in Community.py is what runs the model once we initialize an instance of a community object which automatically sets the parameters except for the initial conditions and supply rate.
setInitialConditions in Community.py sets n0 (initial abundance),c0 (initial resource conc.),a0 (initial traits) and s (supply rate). Can be done manually (there's also many variations of this method that could be useful). If seting initial conditions manually just make sure to concatenate them into z0 (merged (n0,c0,a0,E0) which is the input to odeint which solves the equations.
