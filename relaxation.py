import numpy as np
import copy

np.seterr(divide='ignore', invalid='ignore')

np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)


def dual(primal):
    shape = primal[:-1].shape
    constraints = shape[0]
    decision_variables = shape[1] - 2 - constraints
    slackless = primal[:constraints].T[:decision_variables]
    slack_variables = np.identity(slackless.shape[0])
    z_column = np.matrix(np.zeros(slackless.shape[0])).T
    objective = primal[-1].T[:decision_variables]
    dual = np.concatenate((slackless, slack_variables, z_column, objective),
                          axis=1)
    objective_function = primal[:-1].T[-1]
    objective_slack = np.matrix(np.zeros(slack_variables.shape[0]))
    objective_function = np.concatenate(
        (objective_function, objective_slack, np.matrix([[1, 0]])), axis=1)
    dual = np.concatenate((dual, objective_function))
    return dual


def generate_cut(matrix, cut_matrix):
    # z_row cut
    z_row = matrix[-1]
    lhs = z_row.T[:-1].T
    int_lhs = lhs.astype(int)
    float_lhs = lhs - int_lhs
    rhs = z_row.T[-1].T
    int_rhs = rhs.astype(int)
    float_rhs = rhs - int_rhs
    new_constraint = -1 * np.concatenate((float_lhs, float_rhs), axis=1)
    new_constraint = np.insert(new_constraint, [-2], [1], axis=1)

    # Insert new slack variable into matrix
    slack_column = np.matrix(cut_matrix.shape[0] * [0])
    cut_matrix = np.insert(cut_matrix, [-2], slack_column.T, axis=1)

    # insert new constraint into matrix
    cut_matrix = np.insert(cut_matrix, [-1], new_constraint, axis=0)

    return cut_matrix


def check_cut(matrix):
    return False


def find_pivot_column(
        matrix: int,
) -> int:
    # returns the column index of the lowest value
    # under 0 of the objective function in the matrix
    return int(np.where(matrix[-1] == np.amin(matrix[-1]))[1])


def find_pivot_row(
        matrix: int,
        pivot_column: int,
) -> list[int, bool]:
    # remove objective function from matrix
    masked_matrix = matrix[:-1]

    # divide right hand side of masked matrix
    # by pivot column element for each row
    division_column = masked_matrix.T[-1] / masked_matrix.T[pivot_column]

    # flatten to array
    division_column = np.squeeze(np.asarray(division_column))

    # find indices of all non-nan and values > 0 in array
    non_zero_values = np.where(division_column > 0)[0]

    # return false if no values found above 0
    if len(non_zero_values) < 1:
        return [0, False]
    else:
        # find lowest non-zero value of this division
        pivot_row_index = non_zero_values[
            division_column[non_zero_values].argmin()]
        return [pivot_row_index, True]


def find_pivot_element(
        matrix,
) -> dict:
    # Find pivot element: lowest non-zero
    # Find index of pivot column
    pivot_column_index = find_pivot_column(matrix)
    pivot_row_index, feasible = find_pivot_row(matrix, pivot_column_index)
    if feasible:
        pass
    else:
        print("Simplex infeasible.")
        return False
    return {'column': pivot_column_index, 'row': pivot_row_index}


def simplex(matrix):
    state = np.any(matrix[-1] < 0)
    # PIVOT
    while state:
        pivot_element = find_pivot_element(matrix)
        if type(pivot_element) != dict:
            return matrix
        else:
            # Transform pivot element to 1 by row operation
            pivot_row = matrix[pivot_element['row']] / matrix[
                pivot_element['row'], pivot_element['column']]
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
                check_zeros = check_matrix.T[pivot_element['column']].any(
                    axis=1)

            matrix = np.insert(check_matrix, pivot_element['row'], pivot_row,
                               0)

            # Check if state changed
            state = np.any(matrix[-1] < 0)
            # print(matrix.round(decimals=2))
    return matrix


def branch_and_cut(
        matrix: int,
) -> int:
    # Check if cuts are possible
    # Apply cuts and resolve with simplex
    cut = check_cut(matrix)
    cut_matrix = copy.deepcopy(matrix)
    cut = False

    while cut:
        cut_matrix = generate_cut(matrix, cut_matrix)
        dual_cut = dual(cut_matrix)
        dual_solved = simplex(dual_cut)
        matrix = simplex(cut_matrix)
        cut = check_cut(matrix)
    return matrix


def adjust_variables_for_relaxation(
        leq_constraint_list: list,
        relax: bool,
        variable_list: list,
        geq_constraint_list: list,
        current_depth: int,
        objective_value_list: list,
):
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

        # clean up constraints
        counter = 0
        for constraint in leq_cnstr_copy:
            constraint = constraint[0][current_depth + 1:]
            leq_cnstr_copy[counter][0] = constraint
            counter += 1

        counter = 0
        for constraint in geq_constraint_list:
            constraint = constraint[0][current_depth + 1:]
            geq_constraint_list[counter] = constraint
            counter += 1

        # adjust objective value
        counter = 0
        while counter <= current_depth:
            removed_vars.append(objective_value_list[counter])
            counter += 1
        objective_value_list = objective_value_list[current_depth + 1:]

        # Set objective value for static variables
        for var in removed_vars:
            for dvar in variable_list:
                if var[1] == dvar.name:
                    objective_score += dvar.value * var[0]
    else:
        pass
    return leq_cnstr_copy, variable_list, objective_value_list, objective_score, removed_vars


def matrix_set_up(
        leq_cnstr_copy: list,
        objective_value_list: list,

) -> int:
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

    # Return tableau
    return np.matrix(matrix_input, dtype=float)


def read_simplex_output(
        matrix: int,
        objective_score: int,
        objective_value_list: list,
        removed_vars: list,
        variable_list: list,
):
    simplex_result = {}
    simplex_result['objective_value'] = matrix[-1, -1] + objective_score
    count = 0
    for var in objective_value_list:
        column = matrix.T[count]
        if column.sum() == 1 and np.all((column == 0) | (column == 1)):
            row = int(np.where(column == 1)[1])
            dv_value = matrix[row, -1]
        else:
            dv_value = 0
        simplex_result[var[1]] = dv_value
        count += 1
    if len(removed_vars) > 0:
        for var in removed_vars:
            for dvar in variable_list:
                if var[1] == dvar.name:
                    simplex_result[var[1]] = dvar.value
    is_integer_solution = True
    for variable in variable_list:
        if int(simplex_result[variable.name]) == simplex_result[variable.name]:
            pass
        else:
            is_integer_solution = False
    return is_integer_solution, simplex_result

def check_if_relaxed_is_above_best(
        best_found,
        is_integer_solution,
        simplex_result,
        current_depth,
):
    is_promising_branch = True
    if best_found == False:
        return is_promising_branch, is_integer_solution, simplex_result
    elif simplex_result['objective_value'] > best_found:
        return is_promising_branch, is_integer_solution, simplex_result
    else:
        # print(f"Pruned at depth {current_depth}. Simplex result: {simplex_result['objective_value']} lower than best found: {best_found} ")
        is_promising_branch = False
        return is_promising_branch, is_integer_solution, simplex_result

def relaxation(
               current_depth,
               variable_list,
               best_found,
               leq_constraint_list,
               geq_constraint_list,
               objective_value_list,
               relax
):
    (
        leq_cnstr_copy,
        variable_list,
        objective_value_list,
        objective_score,
        removed_vars,
    ) = adjust_variables_for_relaxation(
        leq_constraint_list,
        relax,
        variable_list,
        geq_constraint_list,
        current_depth,
        objective_value_list
    )
    # Set up tableau
    matrix = matrix_set_up(
        leq_cnstr_copy,
        objective_value_list
    )
    # Run simplex algorithm
    matrix = simplex(matrix)
    # Branch and cut
    matrix = branch_and_cut(matrix)
    # Read solution
    (
        is_integer_solution,
        simplex_result
    ) = read_simplex_output(
    matrix,
    objective_score,
    objective_value_list,
    removed_vars,
    variable_list,
    )
    return check_if_relaxed_is_above_best(
        best_found,
        is_integer_solution,
        simplex_result,
        current_depth
    )



