def solve_it(input_data):

    # MIP solver

    counter = 0
    item_list = []
    for line in input_data:
        if counter == 0:
            item_count = int(line.split()[0])
            max_weight = int(line.split()[1])
            print('bp')
            counter +=1
        else:
            new_list = list(line.split())
            new_tuple = (counter,new_list[0],new_list[1])
            item_list.append(new_tuple)
            counter += 1

    # Tuple indices
    # 0 : item number
    # 1 : item value
    # 2 : item weight

    # An integer decision variable has an upper bound and lower bound
    decision_variables = []
    for variable in range(item_count):
        pass










if __name__ == '__main__':

    file_location = '/Users/marnix/Projects/scripts/DO/data/ks_4_0'
    with open(file_location, 'r') as input_data_file:
        solve_it(input_data_file)