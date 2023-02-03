import functions

# Setup
mip = functions.Solver()

# Set mode to minimize or maximize. 1 for max, 0 for min
mip.mode = 1

# Set variables
mip.add_decision_variable(lb=0,ub=10,name='x_1')
mip.add_decision_variable(lb=0,ub=10,name='x_2')
# mip.add_decision_variable(lb=0,ub=10,name='x_3')

# Set objective function
objective_function = '7 * x_1 /+ 10 * x_2'
mip.set_objective_function(objective_function)

# Set constraints
constraint_string = '-1 * x_1/+ 3 * x_2 <= 6'
mip.set_constraint(constraint_string)
constraint_string = '7 * x_1/+1 * x_2 <= 35'
mip.set_constraint(constraint_string)
# constraint_string = '0 * x_1/+1 * x_2 <= 3' /+0 * x_3 <= 3'
# mip.set_constraint(constraint_string)
# constraint_string = '0 * x_1/+0* x_2/+1 * x_3 <= 2'
# mip.set_constraint(constraint_string)

mip.update()
# Solve mip
mip.solve()
mip.report()