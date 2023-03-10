import copy
import relaxation

depth = 0
INITIAL_SIMPLEX = False

class DecisionVariable:
    def __init__(self, lb=int, ub=int, name=str):
        self.lb = lb
        self.ub = ub
        self.name = name
        self.value = 0


class Solver:
    def __init__(self):
        self.variable_list = []
        self.objective_value = 0
        self.objective_value_list = []
        self.leq_constraint_list = []
        self.geq_constraint_list = []
        self.best_found = False
        self.best_solution = 0
        self.mode = 1
        self.initial_mip = []


    def update(self):
        # Upper bound
        for dvar in self.variable_list:
            constraint_string = ''
            for var in self.variable_list:
                if dvar.name == var.name:
                    constraint_string += f'+1*{var.name}/'
                else:
                    constraint_string += f'+0*{var.name}/'
            constraint_string = constraint_string[:-1]
            constraint_string += f'<= {dvar.ub}'
            self.set_constraint(constraint_string)
        # Lower bound
        for dvar in self.variable_list:
            constraint_string = ''
            for var in self.variable_list:
                if dvar.name == var.name:
                    constraint_string += f'-1*{var.name}/'
                else:
                    constraint_string += f'+0*{var.name}/'
            constraint_string = constraint_string[:-1]
            constraint_string += f'<= {dvar.lb}'
            self.set_constraint(constraint_string)
        print('Added constraints for lower and upper bounds of decision variables')


    def add_decision_variable(self,lb=int,ub=int,name=str):
        new_var = DecisionVariable(lb=lb,ub=ub,name=name)
        self.variable_list.append(new_var)


    def set_objective_function(self,string):
        # Set objective value by separating additions or substractions by a / symbol
        # Example: '-3 * x_1 / + 4 * x_2 / -3 * x_3'
        string = string.replace(' ', '')
        string_list = list(string.split('/'))
        counter = 0

        for item in string_list:
            product_list = list(item.split('*'))
            self.objective_value_list.append(product_list)

        for element in self.objective_value_list:
            self.objective_value_list[counter] = [int(element[0]), element[1]]
            counter += 1


    def set_constraint(self,string):
        self.new_constraint_list = []
        self.new_left_list = []
        self.new_right_list = []
        # Set constraint by separating additions or subtractions by a / symbol

        # Check if constraint is greater than or smaller than:
        if '<' in string:
            left_side,right_side = string.split('<')

            if '=' in string:
                right_side = right_side.replace('=','')
                right_side = int(right_side)
            else:
                right_side = int(right_side) -1
        else:
            left_side,right_side = string.split('>')
            if '=' in string:
                right_side = right_side.replace('=', '')
                right_side = int(right_side)
            else:
                right_side = int(right_side) + 1

        self.new_right_list.append(right_side)

        # Format string to convert it to a form
        # that can be used to calculate if the constraint holds:
        left_side = left_side.replace(' ', '')
        string_list = list(left_side.split('/'))
        counter = 0

        for item in string_list:
            product_list = list(item.split('*'))
            self.new_left_list.append(product_list)

        for element in self.new_left_list:
            self.new_left_list[counter] = [int(element[0]), element[1]]
            counter += 1

        # Combine left and right side of inequality
        self.new_constraint_list.append(self.new_left_list)
        self.new_constraint_list.append(self.new_right_list)

        # add to correct constraint list
        if '<' in string:
            self.leq_constraint_list.append(self.new_constraint_list)
        else:
            self.geq_constraint_list.append(self.new_constraint_list)


    def calculate_objective_value(self,var_list):
        calculated_value = 0
        for var in var_list:
            for ob_var in self.objective_value_list:
                if var.name == ob_var[1]:
                    calculated_value += var.value * ob_var[0]
        return calculated_value


    def check_constraints(self):
        for constraint in self.geq_constraint_list:
            left_side_value = 0
            left = constraint[0]
            right = constraint[1]
            for var in self.variable_list:
                for constraint_var in left:
                    if var.name == constraint_var[1]:
                        left_side_value += var.value * constraint_var[0]
            if left_side_value > right[0]:
                pass
            else:
                return False

        for constraint in self.leq_constraint_list:
            left_side_value = 0
            left = constraint[0]
            right = constraint[1]
            for var in self.variable_list:
                for constraint_var in left:
                    if var.name == constraint_var[1]:
                        left_side_value += var.value * constraint_var[0]
            if left_side_value <= right[0]:
                pass
            else:
                return False
        return True


    def check_max(self):
        if self.mode == 1 and self.best_found != False:
            if self.calculate_objective_value(self.best_solution) < self.calculate_objective_value(self.variable_list):
                self.best_solution = copy.deepcopy(self.variable_list)
                self.best_found = self.calculate_objective_value(self.best_solution)
        elif self.mode == 0 and self.best_found != False:
            if self.calculate_objective_value(self.best_solution) > self.calculate_objective_value(self.variable_list):
                self.best_solution = copy.deepcopy(self.variable_list)
                self.best_found = self.calculate_objective_value(self.best_solution)
        else:
            self.best_solution = copy.deepcopy(self.variable_list)
            self.best_found = self.calculate_objective_value(self.best_solution)


    def solve(self):
        # Run simplex solver to check if it outputs an integer solution
        (
            is_promising_branch,
            is_integer_solution,
            simplex_result,
        ) = relaxation.relaxation(
            0,
            self.variable_list,
            self.best_found,
            self.leq_constraint_list,
            self.geq_constraint_list,
            self.objective_value_list,
            False
        )
        if is_integer_solution and INITIAL_SIMPLEX:
            self.check_if_initial_simplex_solved_mip(simplex_result)
        else:
            # Continue with branch and bound if no integer solution has been found with simplex
            self.branch_and_bound()


    def check_if_initial_simplex_solved_mip(self, simplex_result: int):
        print('\nOptimal integer solution found with initial simplex solve')
        self.best_solution = []
        for dvar in self.variable_list:
            for key in simplex_result:
                if dvar.name == key:
                    dvar.value = simplex_result[key]
                    self.best_solution.append(dvar)
        self.best_found = simplex_result['objective_value']
        if self.check_constraints():
            self.check_max()
        else:
            pass


    def report(self):
        output_list = []
        for var in self.best_solution:
            output_list.append((var.name,var.value))
        print(f'\nBest found solution:{output_list }')
        print(f'Objective value: {self.best_found}')


    def branch_and_bound(self):
        self.best_solution = copy.deepcopy(self.variable_list[:])
        self.dive(depth)

    def reset_lower_nodes_to_zero(self,
        current_depth: int,
    ):
        for lower_node in range(current_depth + 1, len(self.variable_list)):
            self.variable_list[lower_node].value = 0

    def dive(self, depth: int):
        current_depth = depth
        # Loop over all states within lower and upper bound of decision variable
        for variable_value in range(self.variable_list[depth].lb, self.variable_list[depth].ub + 1):
            # Set new value of decision variable
            self.variable_list[current_depth].value = variable_value
            # Reset all values of lower nodes to 0
            self.reset_lower_nodes_to_zero(current_depth)

            # Check constraints
            if self.check_constraints():
                (
                    is_promising_branch,
                    is_integer_solution,
                    simplex_result
                ) = relaxation.relaxation(
                current_depth,
                self.variable_list,
                self.best_found,
                self.leq_constraint_list,
                self.geq_constraint_list,
                self.objective_value_list,
                True
                )
                if is_promising_branch:
                    self.check_depth(current_depth)
                else:
                    pass
            else:
                pass


    def check_depth(self,current_depth):
        # Output if at max depth
        if current_depth == len(self.variable_list) - 1:
            self.check_max()
        # Continue if not at max depth
        else:
            new_depth = current_depth + 1
            self.dive(new_depth)









