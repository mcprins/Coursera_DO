import functions

# Setup
mip = functions.Solver()

# Set mode to minimize or maximize. 1 for max, 0 for min
mip.mode = 1

# Set variables
mip.add_decision_variable(lb=0,ub=10,name='x_1')
mip.add_decision_variable(lb=0,ub=10,name='x_2')
mip.add_decision_variable(lb=0,ub=10,name='x_3')

# Set objective function
objective_function = '8 * x_1 /+ 10 * x_2 /+ 7 * x_3'
mip.set_objective_function(objective_function)

# Set constraints
constraint_string = '1 * x_1/+ 3 * x_2/ + 2 * x_3 <= 10'
mip.set_constraint(constraint_string)
constraint_string = '1 * x_1/+5 * x_2/ + 1 * x_3 <= 8'
mip.set_constraint(constraint_string)


mip.update()
# Solve mip
mip.solve()
mip.report()