#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------
#   Local search for TSP
#
#   Author: Shunji Umetani <umetani@ist.osaka-u.ac.jp>
#   Date: 2021/07/02
# --------------------------------------------------------------------

# import modules -----------------------------------------------------
import sys
import time
import math
import argparse
import networkx as netx
import matplotlib.pyplot as plt

# constant -----------------------------------------------------------
OR_OPT_SIZE = 3  # size of sub-path (or_opt_search)

# --------------------------------------------------------------------
#   TSP data
# --------------------------------------------------------------------
class Tsp:
    # constructor ----------------------------------------------------
    def __init__(self):
        self.name = ''  # name of TSP instance
        self.num_node = 0  # number of nodes
        self.coord = []  # coordinate list of nodes

    # read TSP data --------------------------------------------------
    def read(self, args):
        # open file
        input_file = open(args.filename, 'r')
        data = input_file.readlines()
        input_file.close()

        # read data
        for i in range(len(data)):
            data[i] = (data[i].rstrip()).split()
            data[i] = list(filter(lambda str:str != ':', data[i]))  # remove colon
            if len(data[i]) > 0:
                data[i][0] = data[i][0].rstrip(':')
                if data[i][0] == 'NAME':
                    self.name = data[i][1]
                elif data[i][0] == 'TYPE':
                    if data[i][1] != 'TSP':
                        print('Problem type is not TSP!')
                        sys.exit(1)
                elif data[i][0] == 'DIMENSION':
                    self.num_node = int(data[i][1])
                elif data[i][0] == 'EDGE_WEIGHT_TYPE':  # NOTE: accept only EUC_2D
                    if data[i][1] != 'EUC_2D':
                        print('Edge weight type is not EUC_2D')
                        sys.exit(1)
                elif data[i][0] == 'NODE_COORD_SECTION':
                    sec_coord = i

        # coord section
        self.coord = [(0.0, 0.0)] * self.num_node
        line_cnt = sec_coord+1
        for i in range(self.num_node):
            (self.coord)[int(data[line_cnt][0])-1] = (float(data[line_cnt][1]),float(data[line_cnt][2]))
            line_cnt += 1

    # print TSP data -------------------------------------------------
    def write(self):
        print('\n[TSP data]')
        print('name:\t{}'.format(self.name))
        print('#node:\t{}'.format(self.num_node))
        print('coord:\t{}'.format(self.coord))

    # calculate distance (rounded euclidian distance in 2D) ----------
    def dist(self,v1,v2):
        xd = float((self.coord)[v1][0] - (self.coord)[v2][0])
        yd = float((self.coord)[v1][1] - (self.coord)[v2][1])
        return int(math.sqrt(xd * xd + yd * yd)+0.5)


# --------------------------------------------------------------------
#   working data
# --------------------------------------------------------------------
class Work:
    # constructor ----------------------------------------------------
    def __init__(self,tsp):
        self.tour = [i for i in range(tsp.num_node)]  # tour of salesman
        self.obj = self.calc_obj(tsp)  # objective valiue

    # calculate tour length ------------------------------------------
    def calc_obj(self,tsp):
        length = 0
        for i in range(tsp.num_node):
            length += tsp.dist((self.tour)[i],(self.tour)[(i+1) % tsp.num_node])
        return length

    # write WORK data ------------------------------------------------
    def write(self,tsp):
        print('\n[Tour data]')
        print('length= {}'.format(self.calc_obj(tsp)))

    # draw obtained tour ---------------------------------------------
    def draw(self,tsp):
        graph = netx.Graph()
        graph.add_nodes_from([i for i in range(tsp.num_node)])
        coord = {i: ((tsp.coord)[i][0], (tsp.coord)[i][1]) for i in range(tsp.num_node)}
        netx.add_path(graph, self.tour + [(self.tour)[0]])
        netx.draw(graph, coord, with_labels=True)
        plt.axis('off')
        plt.show()

# function -----------------------------------------------------------

# --------------------------------------------------------------------
#   nearest neighbor algorithm
#
#   tsp(I): TSP data
#   work(I/O): working data
# --------------------------------------------------------------------
def nearest_neighbor(tsp, work):
    print('\n[nearest neighbor algorithm]')

    # nearest neighbor
    for i in range(1,tsp.num_node):
        # find nearest unvisited node
        min_dist = float('inf')
        arg_min_dist = None
        for j in range(i,tsp.num_node):
            dist = tsp.dist((work.tour)[i-1],(work.tour)[j])
            if dist < min_dist:
                min_dist = dist
                arg_min_dist = j
        # set nearest unvisited node
        (work.tour)[i], (work.tour)[arg_min_dist] = (work.tour)[arg_min_dist], (work.tour)[i]

    # print tour length
    work.obj = work.calc_obj(tsp)
    print('length= {}'.format(work.obj))


# --------------------------------------------------------------------
#   local search algorithm
#
#   tsp(I): TSP data
#   work(I/O): working data
# --------------------------------------------------------------------
def local_search(tsp, work):
    print('\n[local search algorithm]')
    # local search
    while True:
        # 2-opt neighborhood search
        two_opt_search(tsp, work)
        # Or-opt neighborhood search
        if or_opt_search(tsp, work):
            continue
        # 3-opt neighborhood search
        if three_opt_search(tsp, work):
            continue
        break

    # print tour length
    print('length= {}'.format(work.obj))


# --------------------------------------------------------------------
#   2-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   return: [True] improved
# --------------------------------------------------------------------
def two_opt_search(tsp, work):
    # evaluate difference for 2-opt operation
    def eval_diff(tsp, work, i, j):
        u, next_u = (work.tour)[i], (work.tour)[(i+1) % len(work.tour)]
        v, next_v = (work.tour)[j], (work.tour)[(j+1) % len(work.tour)]
        cur = tsp.dist(u,next_u) + tsp.dist(v,next_v)
        new = tsp.dist(u,v) + tsp.dist(next_u, next_v)
        return new - cur

    # change tour by 2-opt operation
    def change_tour(tsp, work, i, j):
        # update objective value
        work.obj += eval_diff(tsp, work, i, j)
        # reverse sub-path [i+1,...,j]
        (work.tour)[i+1:j+1] = list(reversed((work.tour)[i+1:j+1]))

    # 2-opt neighborhood search
    improved = False
    restart = True
    while restart:
        restart = False
        nbhd = ((i,j)
                for i in range(len(work.tour))
                for j in range(i+2,len(work.tour)))
        for i,j in nbhd:
            # evaluate difference
            delta = eval_diff(tsp, work, i, j)
            if delta < 0:
                # change tour
                change_tour(tsp, work, i, j)
                improved = True
                restart = True
                break
    return improved


# --------------------------------------------------------------------
#   Or-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   size(I): length of subpath
#   return: [True] improved
# --------------------------------------------------------------------
def or_opt_search(tsp, work, size = OR_OPT_SIZE):
    # evaluate difference for Or-opt operation
    def eval_diff(tsp, work, s, i, j):
        head_p, tail_p = (work.tour)[i % len(work.tour)], (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % len(work.tour)], (work.tour)[(i+s) % len(work.tour)]
        v, next_v = (work.tour)[j % len(work.tour)], (work.tour)[(j+1) % len(work.tour)]
        cur = tsp.dist(prev_p,head_p) + tsp.dist(tail_p,next_p) + tsp.dist(v,next_v)
        new_fwd = tsp.dist(prev_p,next_p) + tsp.dist(v,head_p) + tsp.dist(tail_p,next_v)
        new_bak = tsp.dist(prev_p,next_p) + tsp.dist(v,tail_p) + tsp.dist(head_p,next_v)
        if new_fwd <= new_bak:
            return new_fwd - cur, 'fwd'
        else:
            return new_bak - cur, 'bak'

    # change tour by Or-opt operation
    def change_tour(tsp, work, s, i, j, oper):
        # get sub-path [i,...,i+s-1]
        subpath = []
        for h in range(s):
            subpath.append((work.tour)[(i+h) % len(work.tour)])
        if oper == 'bak':
            subpath.reverse()
        # move sub-path [i,...,i+s-1] to j+1
        for h in range(i+s,j+1):
            (work.tour)[(h-s) % len(work.tour)] = (work.tour)[h % len(work.tour)]
        for h in range(s):
            (work.tour)[(j+1-s+h) % len(work.tour)] = subpath[h]
        # update objective value
        work.obj = work.calc_obj(tsp)

    # Or-opt neighborhood search
    improved = False
    restart = True
    while restart:
        restart = False
        nbhd = ((s,i,j)
                for s in range(1,size+1)
                for i in range(len(work.tour))
                for j in range(i+s,i+len(work.tour)-1))
        for s,i,j in nbhd:
            # evaluate difference
            delta, oper = eval_diff(tsp, work, s, i, j)
            if delta < 0:
                # change tour
                change_tour(tsp, work, s, i, j, oper)
                improved = True
                restart = True
                break
    return improved


# --------------------------------------------------------------------
#   3-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   return: [True] improved
# --------------------------------------------------------------------
def three_opt_search(tsp, work):
    # evaluate difference for 3-opt operation
    def eval_diff(tsp, work, i, j, k):
        best, arg_best = float('inf'), None
        u, next_u = (work.tour)[i], (work.tour)[(i+1) % len(work.tour)]
        v, next_v = (work.tour)[j], (work.tour)[(j+1) % len(work.tour)]
        w, next_w = (work.tour)[k], (work.tour)[(k+1) % len(work.tour)]
        cur = tsp.dist(u,next_u) + tsp.dist(v,next_v) + tsp.dist(w,next_w)
        new = tsp.dist(u,next_v) + tsp.dist(v,next_w) + tsp.dist(w,next_u)  # type1
        if new - cur < best:
            best, arg_best = new - cur, 'type1'
        new = tsp.dist(u,w) + tsp.dist(next_v,next_u) + tsp.dist(v,next_w)  # type2
        if new - cur < best:
            best, arg_best = new - cur, 'type2'
        new = tsp.dist(u,next_v) + tsp.dist(w,v) + tsp.dist(next_u,next_w)  # type3
        if new - cur < best:
            best, arg_best = new - cur, 'type3'
        new = tsp.dist(v,u) + tsp.dist(next_w,next_v) + tsp.dist(w,next_u)  # type4
        if new - cur < best:
            best, arg_best = new - cur, 'type4'
        return best, arg_best

    # change tour by 3-opt operation
    def change_tour(tsp, work, i, j, k, oper):
        if oper == 'type1':
            (work.tour)[i+1:k+1] = (work.tour)[j+1:k+1] + (work.tour)[i+1:j+1]
        elif oper == 'type2':
            (work.tour)[i+1:k+1] = list(reversed((work.tour)[j+1:k+1])) + (work.tour)[i+1:j+1]
        elif oper == 'type3':
            (work.tour)[i+1:k+1] = (work.tour)[j+1:k+1] + list(reversed((work.tour)[i+1:j+1]))
        elif oper == 'type4':
            (work.tour)[i+1:k+1] = list(reversed((work.tour)[i+1:j+1])) + list(reversed((work.tour)[j+1:k+1]))
        # update objective value
        work.obj = work.calc_obj(tsp)

    # 3-opt neighborhood search
    improved = False
    restart = True
    while restart:
        restart = False
        nbhd = ((i,j,k)
                for i in range(len(work.tour))
                for j in range(i+2,len(work.tour))
                for k in range(j+2,len(work.tour)))
        for i,j,k in nbhd:
            # evaluate difference
            delta, oper = eval_diff(tsp, work, i, j, k)
            if delta < 0:
                # change tour
                change_tour(tsp, work, i, j, k, oper)
                improved = True
                restart = True
                break
    return improved


# --------------------------------------------------------------------
#   parse arguments
#
#   argv(I): arguments
# --------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser('TSP')
    # input filename of instance
    parser.add_argument('filename', action='store')
    # draw obtained tour
    parser.add_argument('-d', '--draw', action='store_true', help='draw obtained tour')
    return parser.parse_args()


# --------------------------------------------------------------------
#   main
# --------------------------------------------------------------------
def main(argv=sys.argv):
    # parse arguments
    args = parse_args(argv)

    # set starting time
    start_time = time.time()

    # read instance
    tsp = Tsp()
    tsp.read(args)
    tsp.write()

    # solve TSP
    work = Work(tsp)
    nearest_neighbor(tsp, work)  # nearest neighbor algorithm
    local_search(tsp, work)  # local search algorithm
    work.write(tsp)

    # set completion time
    end_time = time.time()

    # display computation time
    print('\nTotal time:\t%.3f sec' % (end_time - start_time))

    # draw obtained tour
    if args.draw == True:
        work.draw(tsp)


# main ---------------------------------------------------------------
if __name__ == "__main__":
    main()

# --------------------------------------------------------------------
#   end of file
# --------------------------------------------------------------------

