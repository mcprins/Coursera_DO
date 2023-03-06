import numpy as np
import copy

def dual(primal):
    shape = primal[:-1].shape
    constraints = shape[0]
    decision_variables = shape[1] - 2 - constraints
    slackless = primal[:constraints].T[:decision_variables]
    slack_variables = np.identity(slackless.shape[0])
    z_column = np.matrix(np.zeros(slackless.shape[0])).T
    objective = primal[-1].T[:decision_variables]
    dual = np.concatenate((slackless, slack_variables,z_column,objective), axis=1)
    objective_function = primal[:-1].T[-1]
    objective_slack = np.matrix(np.zeros(slack_variables.shape[0]))
    objective_function = np.concatenate((objective_function,objective_slack,np.matrix([[1,0]])),axis=1)
    dual = np.concatenate((dual,objective_function))
    return dual

def generate_cut(matrix,cut_matrix):
    # z_row cut
    z_row = matrix[-1]
    lhs = z_row.T[:-1].T
    int_lhs = lhs.astype(int)
    float_lhs = lhs - int_lhs
    rhs = z_row.T[-1].T
    int_rhs = rhs.astype(int)
    float_rhs = rhs - int_rhs
    new_constraint = -1*np.concatenate((float_lhs,float_rhs),axis=1)
    new_constraint = np.insert(new_constraint, [-2], [1], axis=1)


    # Insert new slack variable into matrix
    slack_column = np.matrix(cut_matrix.shape[0] * [0])
    cut_matrix = np.insert(cut_matrix,[-2],slack_column.T,axis=1)

    # insert new constraint into matrix
    cut_matrix = np.insert(cut_matrix,[-1],new_constraint,axis=0)

    return cut_matrix

def check_cut(matrix):
    return False

def find_pivot_element(
        matrix,
    )-> dict:
    # Find pivot element: lowest non-zero
    # Find index of pivot column
    pivot_column = int(np.where(matrix[-1] == np.amin(matrix[-1]))[1])
    masked_matrix = np.delete(matrix,np.where(matrix.T[-1] <= 0),axis=0)
    pivot_row_division = np.divide(masked_matrix.T[-1],
                                   masked_matrix.T[pivot_column])
    masked_pivot_row = pivot_row_division[pivot_row_division > 0]
    try:
        min_row = masked_pivot_row.min()
    except:
        print(f'No legal candidates for pivot. Simplex infeasible.')
        # return True, False, {}

    pivot_row_index = np.where(pivot_row_division == min_row)[1]
    if pivot_row_index.size == 1:
        pivot_row_index = int(pivot_row_index)
    else:
        pivot_row_index = int(pivot_row_index[0])

    return {'column': pivot_column, 'row': pivot_row_index}

def simplex(matrix):
    np.set_printoptions(linewidth=200)
    np.set_printoptions(suppress=True)

    state = np.any(matrix[-1] < 0)
    # PIVOT

    # dual_matrix = dual(matrix)
    # primal_matrix = dual(dual_matrix)
    # print(matrix.round(decimals=2))
    while state:
        pivot_element = find_pivot_element(matrix)

        # Transform pivot element to 1 by row operation
        pivot_row = matrix[pivot_element['row']] / matrix[pivot_element['row'], pivot_element['column']]
        matrix[pivot_element['row']] = pivot_row

        # Transform all elements below and above pivot element to 0 by row operation.
        # Check if elements above and below are 0. If not, repeat.
        check_matrix = np.delete(matrix, pivot_element['row'], 0)
        check_zeros = check_matrix.T[pivot_element['column']].any(axis=1)
        while check_zeros:
            pivot_column = check_matrix.T[pivot_element['column']]
            row_index = np.where(pivot_column != 0)[1][0]
            row_to_change = check_matrix[row_index]
            row_element = float(row_to_change.T[pivot_element['column']])
            pivot_value = float(pivot_row.T[pivot_element['column']])
            multiplication_factor = row_element / pivot_value

            row_to_change -= multiplication_factor * pivot_row
            check_matrix[row_index] = row_to_change
            check_zeros = check_matrix.T[pivot_element['column']].any(axis=1)

        matrix = np.insert(check_matrix, pivot_element['row'], pivot_row, 0)

        # Check if state changed
        state = np.any(matrix[-1] < 0)
        # print(matrix.round(decimals=2))
    return matrix

def relaxation(mode,
                current_depth,
                variable_list,
                best_found,
                leq_constraint_list,
                geq_constraint_list,
                objective_value_list,
                relax):

    leq_cnstr_copy = copy.deepcopy(leq_constraint_list)
    objective_score = 0
    removed_vars = []
    if relax:
        # adjust constraints
        for dvar in variable_list:
            for constraint in leq_cnstr_copy:
                for element in constraint[0]:
                    if dvar.name == element[1]:
                        constraint[1][0] -= dvar.value * element[0]
            # for constraint in geq_constraint_list:
            #     constraint[1][0] -= variable_list[counter].value * constraint[0][counter][0]
        # clean up constraints
        counter = 0
        for constraint in leq_cnstr_copy:
            constraint = constraint[0][current_depth+1:]
            leq_cnstr_copy[counter][0] = constraint
            counter += 1

        counter = 0
        for constraint in geq_constraint_list:
            constraint = constraint[0][current_depth+1:]
            geq_constraint_list[counter] = constraint
            counter += 1

        # adjust objective value
        counter = 0
        while counter <= current_depth:
            removed_vars.append(objective_value_list[counter])
            counter += 1
        objective_value_list = objective_value_list[current_depth+1:]

        # Set objective value for static variables
        for var in removed_vars:
            for dvar in variable_list:
                if var[1] == dvar.name:
                    objective_score += dvar.value * var[0]
    else:
        pass

    if mode == 1:
        # Simplex set-up
        matrix_input = []
        slack_variable_count = len(leq_cnstr_copy)
        constraint_count = 0

        # Rewrite constraints to account for variables that are static because of depth
        for constraint in leq_cnstr_copy:
            new_row = []
            for element in constraint[0]:
                new_row.append(element[0])
            slack_row = slack_variable_count * [0]
            slack_row[constraint_count] = 1
            constraint_count += 1
            new_row = new_row + slack_row + [0] + [constraint[1][0]]
            matrix_input.append(new_row)
        # Rewrite objective function to account for variables that are static because of depth
        objective_value = []
        for element in objective_value_list:
            objective_value.append(-1 * element[0])
        objective_value += slack_variable_count * [0] + [1] + [0]
        matrix_input.append(objective_value)

        # Create initial tableau
        matrix = np.matrix(matrix_input, dtype=float)
        cut_matrix = copy.deepcopy(matrix)
        # Run simplex algorithm
        matrix = simplex(matrix)

        # Check if cuts are possible
        # Apply cuts and resolve with simplex
        # cut = check_cut(matrix)
        cut = False
        # test
        while cut:
            cut_matrix = generate_cut(matrix,cut_matrix)
            dual_cut = dual(cut_matrix)
            dual_solved = simplex(dual_cut)
            matrix = simplex(cut_matrix)
            cut = check_cut(matrix)

        # Read solution
        optimal_solution = {}
        optimal_solution['objective_value'] = matrix[-1, -1] + objective_score
        count = 0
        for var in objective_value_list:
            column = matrix.T[count]
            if column.sum() == 1 and np.all((column == 0) | (column == 1)):
                row = int(np.where(column == 1)[1])
                dv_value = matrix[row,-1]
            else:
                dv_value = 0
            optimal_solution[var[1]] = dv_value
            count += 1
        if len(removed_vars) > 0:
            for var in removed_vars:
                for dvar in variable_list:
                    if var[1] == dvar.name:
                        optimal_solution[var[1]] = dvar.value
        mip = True
        for variable in variable_list:
            if int(optimal_solution[variable.name]) == optimal_solution[variable.name]:
                pass
            else:
                mip = False


        if best_found == False:
            return True,mip,optimal_solution
        elif optimal_solution['objective_value'] > best_found:
            return True,mip,optimal_solution
        else:
            print(f'Pruned at depth {current_depth}')
            return False,mip,optimal_solution
    else:
        pass