#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------
#   Iterated local search for TSP with neighbor-list
#
#   Author: Shunji Umetani <umetani@ist.osaka-u.ac.jp>
#   Date: 2021/07/13
# --------------------------------------------------------------------

# import modules -----------------------------------------------------
import sys
import time
import math
import random
import argparse
import networkx as netx
import matplotlib.pyplot as plt

# constant -----------------------------------------------------------
TIME_LIMIT = 60  # default time limit for iterated local search
INTVL_TIME = 1.0  # interval time for display logs
RANDOM_SEED = 0  # default random seed
ILS_NUM_KICK = 1  # number of kicks in ILS
OR_OPT_SIZE = 3  # size of sub-path (or_opt_search)
NB_LIST_SIZE = 5  # size of neighbor-list

# --------------------------------------------------------------------
#   TSP data
# --------------------------------------------------------------------
class Tsp:
    # constructor ----------------------------------------------------
    def __init__(self):
        self.name = ''  # name of TSP instance
        self.num_node = 0  # number of nodes
        self.coord = []  # coordinate list of nodes
        self.neighbor = []  # neighbor-list

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

    # construct neighbor-list ----------------------------------------
    def gen_neighbor(self):
        self.neighbor = [[] for _ in range(self.num_node)]
        for i in range(self.num_node):
            temp = [(self.dist(i,j),j) for j in range(self.num_node) if j != i]
            temp.sort(key=lambda x: x[0])
            (self.neighbor)[i] = [temp[h][1] for h in range(min(NB_LIST_SIZE,self.num_node))]


# --------------------------------------------------------------------
#   working data
# --------------------------------------------------------------------
class Work:
    # constructor ----------------------------------------------------
    def __init__(self,tsp):
        self.tour = [i for i in range(tsp.num_node)]  # tour of salesman
        self.pos = [i for i in range(tsp.num_node)]  # position of nodes in tour
        self.obj = self.length(tsp)  # objective value
        self.active = [True for _ in range(tsp.num_node)]  # active nodes

    # copy -----------------------------------------------------------
    def copy(self,org):
        self.tour = org.tour[:]
        self.pos = org.pos[:]
        self.obj = org.obj
        self.active = org.active[:]

    # calculate tour length ------------------------------------------
    def length(self,tsp):
        length = 0
        for i in range(len(self.tour)):
            length += tsp.dist((self.tour)[i],(self.tour)[(i+1) % len(self.tour)])
        return length

    # set position ---------------------------------------------------
    def set_pos(self):
        for i in range(len(self.tour)):
            (self.pos)[(self.tour)[i]] = i

    # next node in tour ----------------------------------------------
    def next(self,v):
        return (self.tour)[((self.pos)[v]+1) % len(self.tour)]

    # previous node in tour ------------------------------------------
    def prev(self,v):
        return (self.tour)[((self.pos)[v]-1) % len(self.tour)]

    # write WORK data ------------------------------------------------
    def write(self,tsp):
        print('\n[Tour data]')
        print('length= {}'.format(self.length(tsp)))

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

    # initialize position of nodes in tour
    work.set_pos()
    # calculate tour length
    work.obj = work.length(tsp)

    # print tour length
    print('length= {}'.format(work.obj))


# --------------------------------------------------------------------
#   iterated local search algorithm
#
#   tsp(I): TSP data
#   work(I/O): working data
#   time_limit(I): time_limit
# --------------------------------------------------------------------
def iterated_local_search(tsp, work, time_limit):
    # kick incumbent solution by double-bridge neighborhood operation
    def kick_tour(tsp, work, num_kick = ILS_NUM_KICK):
        for _ in range(num_kick):
            i = random.randrange(0,len(work.tour)-7)
            j = random.randrange(i+2,len(work.tour)-5)
            k = random.randrange(j+2,len(work.tour)-3)
            l = random.randrange(k+2,len(work.tour)-1)
            (work.tour)[i+1:l+1] = (work.tour)[k+1:l+1] + (work.tour)[j+1:k+1] + (work.tour)[i+1:j+1]
            work.obj = work.length(tsp)
            # initialize position of nodes in tour
            work.set_pos()
            # activate nodes
            for v in work.tour:
                (work.active)[v] = False
            u = (work.tour)[i]
            (work.active)[u] = (work.active)[work.next(u)] = True
            for y in (tsp.neighbor)[u]:
                (work.active)[y] = True
            for y in (tsp.neighbor)[work.next(u)]:
                (work.active)[y] = True
            v = (work.tour)[j]
            (work.active)[v] = (work.active)[work.next(v)] = True
            for y in (tsp.neighbor)[v]:
                (work.active)[y] = True
            for y in (tsp.neighbor)[work.next(v)]:
                (work.active)[y] = True
            w = (work.tour)[k]
            (work.active)[w] = (work.active)[work.next(w)] = True
            for y in (tsp.neighbor)[w]:
                (work.active)[y] = True
            for y in (tsp.neighbor)[work.next(w)]:
                (work.active)[y] = True
            x = (work.tour)[l]
            (work.active)[x] = (work.active)[work.next(x)] = True
            for y in (tsp.neighbor)[x]:
                (work.active)[y] = True
            for y in (tsp.neighbor)[work.next(x)]:
                (work.active)[y] = True

    # iterated local search
    print('\n[iterated local search algorithm]')

    # initialize current working data
    cur_work = Work(tsp)
    cur_work.copy(work)

    # initialize active nodes
    cur_work.active = [True for _ in range(tsp.num_node)]

    # iterated local search
    start_time = cur_time = disp_time = time.time()
    cnt = 0
    while cur_time - start_time < time_limit:
        # kick incumbent tour
        cur_work.copy(work)
        kick_tour(tsp, cur_work)
        # local search algorithm
        local_search(tsp, cur_work)
        cnt += 1
        cur_time = time.time()
        # update best working data
        if cur_work.obj < work.obj:
            work.copy(cur_work)
            print('{}\t{}*\t{}\t{:.2f}'.format(cnt,cur_work.obj,work.obj,cur_time-start_time))
        elif cur_time - disp_time > INTVL_TIME:
            print('{}\t{}\t{}\t{:.2f}'.format(cnt,cur_work.obj,work.obj,cur_time-start_time))
            disp_time = time.time()

    # print tour length
    print('length= {}'.format(work.obj))


# --------------------------------------------------------------------
#   local search algorithm
#
#   tsp(I): TSP data
#   work(I/O): working data
# --------------------------------------------------------------------
def local_search(tsp, work):
    # local search
    restart = True
    while restart:
        restart = False
        for u in work.tour:
            if (work.active)[u]:
                # 2-opt neighborhood search
                if two_opt_search(tsp, work, u):
                    restart = True
                    break
                # Or-opt neighborhood search
                if or_opt_search(tsp, work, u):
                    restart = True
                    break
                # 3-opt neighborhood search
                if three_opt_search(tsp, work, u):
                    restart = True
                    break
            # inactivate node u
            (work.active)[u] = False


# --------------------------------------------------------------------
#   2-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   u(I): pivot node
#   return: [True] improved
# --------------------------------------------------------------------
def two_opt_search(tsp, work, u):
    # evaluate difference for 2-opt operation
    def eval_diff(tsp, work, u, v):
        cur = tsp.dist(u,work.next(u)) + tsp.dist(v,work.next(v))
        new = tsp.dist(u,v) + tsp.dist(work.next(u),work.next(v))
        return new - cur

    # change tour by 2-opt operation
    def change_tour(tsp, work, u, v):
        if (work.pos)[u] < (work.pos)[v]:
            i, j = (work.pos)[u], (work.pos)[v]
        else:
            i, j = (work.pos)[v], (work.pos)[u]
        # reverse sub-path [i+1,...,j]
        (work.tour)[i+1:j+1] = list(reversed((work.tour)[i+1:j+1]))
        # update positions
        work.set_pos()
        # update objective value
        work.obj = work.length(tsp)

    # 2-opt neighborhood search
    for v in (tsp.neighbor)[u]:
        # evaluate difference
        delta = eval_diff(tsp, work, u, v)
        if delta < 0:
            # activate nodes
            (work.active)[u] = (work.active)[work.next(u)] = True
            (work.active)[v] = (work.active)[work.next(v)] = True
            # change current tour
            change_tour(tsp, work, u, v)
            return True
    return False


# --------------------------------------------------------------------
#   Or-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   u(I): pivot node
#   size(I): length of subpath
#   return: [True] improved
# --------------------------------------------------------------------
def or_opt_search(tsp, work, u, size = OR_OPT_SIZE):
    # evaluate difference for Or-opt operation
    def eval_diff(tsp, work, s, u, v):
        i = (work.pos)[u]
        head_p, tail_p = u, (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % tsp.num_node], (work.tour)[(i+s) % len(work.tour)]
        # forward insertion
        cur = tsp.dist(prev_p,head_p) + tsp.dist(tail_p,next_p) + tsp.dist(v,work.next(v))
        new = tsp.dist(prev_p,next_p) + tsp.dist(v,head_p) + tsp.dist(tail_p,work.next(v))
        fwd_diff = new - cur
        # check node v in sub-path
        for h in range(-1,s):
            if v == (work.tour)[(i+h) % len(work.tour)]:
                fwd_diff = float('inf')
        # backward insertion
        cur = tsp.dist(prev_p,head_p) + tsp.dist(tail_p,next_p) + tsp.dist(work.prev(v),v)
        new = tsp.dist(prev_p,next_p) + tsp.dist(work.prev(v),tail_p) + tsp.dist(head_p,v)
        bak_diff = new - cur
        # check node prev_v in sub-path
        for h in range(-1,s):
            if work.prev(v) == (work.tour)[(i+h) % len(work.tour)]:
                bak_diff = float('inf')
        if fwd_diff <= bak_diff:
            return fwd_diff, 'fwd'
        else:
            return bak_diff, 'bak'

    # change tour by Or-opt operation
    def change_tour(tsp, work, s, u, v, oper):
        pop_pos = (work.pos)[u]
        if oper == 'fwd':
            ins_pos = ((work.pos)[v]+1) % len(work.tour)
        else:
            ins_pos = (work.pos)[v]
        # get sub-path
        subpath = []
        for h in range(s):
            subpath.append((work.tour)[(pop_pos+h) % len(work.tour)])
        if oper == 'bak':
            subpath.reverse()
        # move sub-path [i,...,i+s-1] to j+1 (forward) or j (backward)
        if pop_pos > ins_pos:
            for h in range(pop_pos+s,ins_pos+len(work.tour)):
                (work.tour)[(h-s) % len(work.tour)] = (work.tour)[h % len(work.tour)]
        else:
            for h in range(pop_pos+s,ins_pos):
                (work.tour)[(h-s) % len(work.tour)] = (work.tour)[h % len(work.tour)]
        for h in range(s):
            (work.tour)[(ins_pos-s+h) % len(work.tour)] = subpath[h]
        # update positions
        work.set_pos()
        # update objective value
        work.obj = work.length(tsp)

    # activate nodes
    def activate_node(work, s, u, v):
        i = (work.pos)[u]
        head_p, tail_p = u, (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % tsp.num_node], (work.tour)[(i+s) % len(work.tour)]
        (work.active)[head_p] = (work.active)[tail_p] = True
        (work.active)[prev_p] = (work.active)[next_p] = True
        (work.active)[v] = (work.active)[work.next(v)] = True

    # Or-opt neighborhood search
    nbhd = ((s,v)
            for s in range(1,size+1)
            for v in (tsp.neighbor)[u])
    for s,v in nbhd:
        # evaluate difference
        delta, oper = eval_diff(tsp, work, s, u, v)
        if delta < 0:
            # activate nodes
            activate_node(work, s, u, v)
            # change current tour
            change_tour(tsp, work, s, u, v, oper)
            return True
    return False


# --------------------------------------------------------------------
#   3-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   u(I): pivot node
#   return: [True] improved
# --------------------------------------------------------------------
def three_opt_search(tsp, work, u):
    # evaluate difference for 3-opt operation
    def eval_diff_type134(tsp, work, u, v, w):
        best, arg_best = float('inf'), None
        # type1
        cur = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(w,work.next(w))
        new = tsp.dist(u,v) + tsp.dist(work.prev(v),work.next(w)) + tsp.dist(w,work.next(u))
        if new - cur < best and (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+1:  # check node v and w
            best, arg_best = new - cur, 'type1'
        # type3
        cur = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(work.prev(w),w)
        new = tsp.dist(u,v) + tsp.dist(work.prev(w),work.prev(v)) + tsp.dist(work.next(u),w)
        if new - cur < best and (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+2:  # check node v and w
            best, arg_best = new - cur, 'type3'
        # type4
        cur = tsp.dist(u,work.next(u)) + tsp.dist(v,work.next(v)) + tsp.dist(w,work.next(w))
        new = tsp.dist(u,v) + tsp.dist(work.next(u),w) + tsp.dist(work.next(v),work.next(w))
        if new - cur < best and (work.pos)[v] >= (work.pos)[u]+2 and (work.pos)[w] >= (work.pos)[v]+2:  # check node v and w
            best, arg_best = new - cur, 'type4'
        return best, arg_best

    def eval_diff_type2(tsp, work, u, v, w):
        cur = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(w,work.next(w))
        new = tsp.dist(u,w) + tsp.dist(v,work.next(u)) + tsp.dist(work.prev(v),work.next(w))
        if (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+1:
            return new - cur, 'type2'
        else:
            return float('inf'), None

    # change tour by 3-opt operation
    def change_tour(tsp, work, u, v, w, oper):
        i,j,k = (work.pos)[u], (work.pos)[v],(work.pos)[w]
        if oper == 'type1':
            (work.tour)[i+1:k+1] = (work.tour)[j:k+1] + (work.tour)[i+1:j]
        elif oper == 'type2':
            (work.tour)[i+1:k+1] = list(reversed((work.tour)[j:k+1])) + (work.tour)[i+1:j]
        elif oper == 'type3':
            (work.tour)[i+1:k] = (work.tour)[j:k] + list(reversed((work.tour)[i+1:j]))
        elif oper == 'type4':
            (work.tour)[i+1:k+1] = list(reversed((work.tour)[i+1:j+1])) + list(reversed((work.tour)[j+1:k+1]))
        # update positions
        work.set_pos()
        # update objective value
        work.obj = work.length(tsp)

    # activate nodes
    def activate_node(work, u, v, w, oper):
        if oper == 'type1' or oper == 'type2':
            (work.active)[u] = (work.active)[work.next(u)] = True
            (work.active)[v] = (work.active)[work.prev(v)] = True
            (work.active)[w] = (work.active)[work.next(w)] = True
        elif oper == 'type3':
            (work.active)[u] = (work.active)[work.next(u)] = True
            (work.active)[v] = (work.active)[work.prev(v)] = True
            (work.active)[w] = (work.active)[work.prev(w)] = True
        elif oper == 'type4':
            (work.active)[u] = (work.active)[work.next(u)] = True
            (work.active)[v] = (work.active)[work.next(v)] = True
            (work.active)[w] = (work.active)[work.next(w)] = True

    # 3-opt neighborhood search
    nbhd = ((v,w)
            for v in (tsp.neighbor)[u]
            for w in (tsp.neighbor)[work.next(u)])
    for v,w in nbhd:
        # evaluate difference
        delta, oper = eval_diff_type134(tsp, work, u, v, w)
        if delta < 0:
            # activate nodes
            activate_node(work, u, v, w, oper)
            # change current tour
            change_tour(tsp, work, u, v, w, oper)
            return True
    nbhd = ((v,w)
            for v in (tsp.neighbor)[work.next(u)]
            for w in (tsp.neighbor)[u])
    for v,w in nbhd:
        # evaluate difference
        delta, oper = eval_diff_type2(tsp, work, u, v, w)
        if delta < 0:
            # activate nodes
            activate_node(work, u, v, w, oper)
            # change current tour
            change_tour(tsp, work, u, v, w, oper)
            return True
    return False


# --------------------------------------------------------------------
#   parse arguments
#
#   argv(I): arguments
# --------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser('TSP')
    # input filename of instance
    parser.add_argument('filename', action='store')
    # timelimit for solver
    parser.add_argument('-t', '--time', help='time limit for iterated local search', type=float, default=TIME_LIMIT)
    # draw obtained tour
    parser.add_argument('-d', '--draw', action='store_true', help='draw obtained tour')
    return parser.parse_args()


# --------------------------------------------------------------------
#   main
# --------------------------------------------------------------------
def main(argv=sys.argv):
    # parse arguments
    args = parse_args(argv)

    # set random seed
    random.seed(RANDOM_SEED)

    # set starting time
    start_time = time.time()

    # read instance
    tsp = Tsp()
    tsp.read(args)
    tsp.write()

    # construct neighbor-list
    tsp.gen_neighbor()

    # solve TSP
    work = Work(tsp)
    nearest_neighbor(tsp, work)  # nearest neighbor algorithm
    iterated_local_search(tsp, work, args.time)  # iterated local search
    work.write(tsp)

    # set completion time
    end_time = time.time()

    # display computaion time
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
