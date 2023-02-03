#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from operator import itemgetter
import sys
import timeit

sys.setrecursionlimit(1000000)

Item = namedtuple("Item",['index','value','weight','density','taken'])

def solve_it(input_data):

    #################
    # INPUT PARSING #
    #################

    lines = input_data.split('\n')
    progress_tik = timeit.default_timer()

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(
            Item(
                i-1,
                int(parts[0]),
                int(parts[1]),
                int(parts[0])/int(parts[1]),
                0
            )
        )
    # Sort items list by value density
    items = sorted(items, key=itemgetter(3), reverse=True)

    # Set up variables for use in solver
    depth = 0
    taken = [0] * len(items)

    class BranchAndBound():
        def __init__(self):
            self.max_value = 0
            self.local_taken = taken
            self.items_listed = []
            self.best_found_taken = []

            # Convert list of tuples to list of lists
            for index, item in enumerate(items):
                listed_tuple = list(item)
                self.items_listed.append(listed_tuple)

        def calculate_progress(self,local_taken):
            progress = 0
            for count, binary_value in enumerate(local_taken):
                if binary_value == 0:
                    progress += 50 / 2 ** count
            return progress

        # Taken weight and value function
        def calculate_weight_value(self,item_list,depth):
            self.taken_weight = 0
            self.taken_value = 0
            for i, j, k in zip(item_list, self.local_taken, [x for x in range(depth+1)]):
                self.taken_weight += i[2] * j
                self.taken_value += i[1] * j
            return self.taken_weight,self.taken_value

        # Relaxed optimum
        def relaxed_optimum(self,depth):
            # Create variables
            self.item_list = []

            # Convert list of tuples to list of lists
            for index, item in enumerate(items):
                listed_tuple = list(item)
                self.item_list.append(listed_tuple)

            # Calculate the weight and value of the items that have already been chosen
            self.taken_weight, self.taken_value = self.calculate_weight_value(self.item_list,depth)

            # Remove the items from the list of lists that have already been chosen
            for i in range(depth):
                 self.item_list.pop(0)

            # Sort the list based on value density
            self.item_list = sorted(self.item_list, key=itemgetter(3), reverse=True)

            # Add the highest density item to taken list until capacity constraint is reached
            for i in self.item_list:
                if capacity - self.taken_weight >= i[2]:
                    self.taken_weight += i[2]
                    self.taken_value += i[1]

            # Check if a fraction of an item can be added
                elif (capacity - self.taken_weight) > 0 and i[2] > (capacity - self.taken_weight):
                    self.taken_value += i[1] * ((float(capacity) - float(self.taken_weight)) / float(i[2]))
                    self.taken_weight = capacity
            return int(self.taken_value)

        def status_output(self):
            progress_tok = timeit.default_timer()
            self.gap = 100 * (1 - (self.max_value / self.relaxed_optimum(0)))
            print("***********************")
            print("New maximum:     " + str(self.max_value))
            print("Gap:             " + str(round(self.gap, 3)) + " %")
            print("Search progress: " + str(round(self.calculate_progress(self.local_taken), 3)) + " %")
            print("Elapsed time:    " + str(round(progress_tok-progress_tik,1)))

        def update_best_found(self):
            self.max_value = self.current_value
            for i,j in zip(self.local_taken,self.items_listed):
                j[4] = i
            self.status_output()




        def check_depth(self, current_depth):
            self.current_weight, self.current_value = self.calculate_weight_value(items, current_depth)
            if self.relaxed_optimum(depth) < self.max_value:
                 pass
            else:
                if self.current_weight <= capacity:
                    if current_depth == len(items) - 1:
                        if self.current_value > self.max_value:
                            self.update_best_found()
                    else:
                        new_depth = current_depth + 1
                        self.recursive_bb(new_depth)

        def sort_index(self):
            self.items_listed = sorted(self.items_listed, key=itemgetter(0), reverse=False)
            for i in self.items_listed:
                self.best_found_taken.append(i[4])


        # BRANCH AND BOUND #

        def recursive_bb(self,depth):
            progress_tok = timeit.default_timer()
            if int(progress_tok) - int(progress_tik) < 180:
                current_depth = depth
            # Take item
                if current_depth <= len(items)-1:
                    self.local_taken[current_depth] = 1
                    self.check_depth(current_depth)
            # Do not take item
                    self.local_taken[current_depth] = 0
                    self.check_depth(current_depth)

        def greedy_algorithm(self):
            weight = 0
            for item in self.items_listed:
                if weight + item[2] <= capacity:
                    item[4] = 1
                    self.max_value += item[1]
                    weight += item[2]

    tik = timeit.default_timer()
    bb = BranchAndBound()
    if len(items) > 1000:
        bb.greedy_algorithm()
    else:
        bb.recursive_bb(depth)
    bb.sort_index()
    tok = timeit.default_timer()
    print("-----------------------")
    print("Solve time: " + str(round(tok-tik,3)))
    taken = bb.best_found_taken
    value = bb.max_value


    #######################
    # OUTPUT PREPARATION  #
    #######################

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')