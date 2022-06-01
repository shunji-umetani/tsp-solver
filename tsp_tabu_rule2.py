#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------
#   Tabu search for TSP
#
#   Author: Shunji Umetani <umetani@ist.osaka-u.ac.jp>
#   Date: 2022/05/03
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
TIME_LIMIT = 60  # default time limit for multi-start local search
RANDOM_SEED = 0  # default random seed
OR_OPT_SIZE = 3  # size of sub-path (or_opt_search)
NB_LIST_SIZE = 5  # size of neighbor-list
TABU_TENURE_RATIO = 1  # tabu tenure ratio
INTVL_TIME = 1.0  # interval time for display logs

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
        self.cnt = 0  # counter
        self.move = {}  # count of last updated

    # copy -----------------------------------------------------------
    def copy(self,org):
        self.tour = org.tour[:]
        self.pos = org.pos[:]
        self.obj = org.obj
        self.cnt = org.cnt
        self.move = (org.move).copy()

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

    # check tabu list ------------------------------------------------
    def get_tabu(self, u, v):
        if u > v:
            u,v = v,u
        if (u,v) in self.move and self.cnt <= (self.move)[(u,v)]:
            return True
        else:
            return False

    # set tabu list --------------------------------------------------
    def set_tabu(self, u, v):
        if u <= v:
            (self.move)[(u,v)] = self.cnt + random.randint(1,int(TABU_TENURE_RATIO * math.sqrt(NB_LIST_SIZE * len(self.tour)))+1)
        else:
            (self.move)[(v,u)] = self.cnt + random.randint(1,int(TABU_TENURE_RATIO * math.sqrt(NB_LIST_SIZE * len(self.tour)))+1)

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
#   Tabu search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   time_limit(I): time_limit
# --------------------------------------------------------------------
def tabu_search(tsp, work, time_limit):
    print('\n[tabu search algorithm]')

    # initialize current working data
    cur_work = Work(tsp)
    cur_work.copy(work)

    # tabu search
    best_obj = float('inf')
    start_time = cur_time = disp_time = time.time()
    while cur_time - start_time < time_limit:
        while True:
            # 2-opt neighborhood search
            if two_opt_search(tsp, work, cur_work):
                continue
            # Or-opt neighborhood search
            if or_opt_search(tsp, work, cur_work):
                continue
            # 3-opt neighborhood search
            if three_opt_search(tsp, work, cur_work):
                continue
            break
        cur_time = time.time()
        if work.obj < best_obj:
            print('{}\t{}\t{}*\t{:.2f}'.format(cur_work.cnt,cur_work.obj,work.obj,cur_time - start_time))
            best_obj = work.obj
        elif cur_time - disp_time >= INTVL_TIME:
            print('{}\t{}\t{}\t{:.2f}'.format(cur_work.cnt,cur_work.obj,work.obj,cur_time - start_time))
            disp_time = cur_time


# --------------------------------------------------------------------
#   2-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   cur_work(I/O): current working data
#   return: [True] improved
# --------------------------------------------------------------------
def two_opt_search(tsp, work, cur_work):
    # evaluate difference for 2-opt operation
    def eval_diff(tsp, work, u, v):
        cur = tsp.dist(u,work.next(u)) + tsp.dist(v,work.next(v))
        new = tsp.dist(u,v) + tsp.dist(work.next(u),work.next(v))
        return new - cur

    # check tabu (add edges)
    def check_tabu(work, u, v):
        if work.get_tabu(u,v) or work.get_tabu(work.next(u),work.next(v)):
            return True
        else:
            return False

    # update tabu-list (delete edges)
    def update_tabu(work, u, v):
        work.set_tabu(u, work.next(u))
        work.set_tabu(v, work.next(v))

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
    min_delta, arg_min_delta = float('inf'), None
    nbhd = ((u,v)
            for u in cur_work.tour
            for v in (tsp.neighbor)[u])
    for u,v in nbhd:
        asp_flag = False
        # evaluate difference
        delta = eval_diff(tsp, cur_work, u, v)
        # update incumbent solution
        if cur_work.obj + delta < work.obj:
            asp_flag = True
            work.copy(cur_work)
            change_tour(tsp, work, u, v)
        # update best solution in the neighborhood
        if delta < min_delta and (not check_tabu(cur_work,u,v) or asp_flag):
            min_delta = delta
            arg_min_delta = (u,v)
    if arg_min_delta is not None:
        # update tabu-list
        update_tabu(cur_work, *arg_min_delta)
        # update current working data
        change_tour(tsp, cur_work, *arg_min_delta)
    cur_work.cnt += 1
    if min_delta < 0:
        return True
    else:
        return False


# --------------------------------------------------------------------
#   Or-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   cur_work(I/O): current working data
#   size(I): length of subpath
#   return: [True] improved
# --------------------------------------------------------------------
def or_opt_search(tsp, work, cur_work, size = OR_OPT_SIZE):
    # evaluate difference for Or-opt operation
    def eval_diff(tsp, work, s, u, v):
        i = (work.pos)[u]
        head_p, tail_p = u, (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % tsp.num_node], (work.tour)[(i+s) % len(work.tour)]
        # forward insertion
        cur_fwd = tsp.dist(prev_p,head_p) + tsp.dist(tail_p,next_p) + tsp.dist(v,work.next(v))
        new_fwd = tsp.dist(prev_p,next_p) + tsp.dist(v,head_p) + tsp.dist(tail_p,work.next(v))
        diff_fwd = new_fwd - cur_fwd
        # check node v in subpath
        for h in range(-1,s):
            if v == (work.tour)[(i+h) % len(work.tour)]:
                diff_fwd = float('inf')
        # backward insertion
        cur_bak = tsp.dist(prev_p,head_p) + tsp.dist(tail_p,next_p) + tsp.dist(work.prev(v),v)
        new_bak = tsp.dist(prev_p,next_p) + tsp.dist(work.prev(v),tail_p) + tsp.dist(head_p,v)
        diff_bak = new_bak - cur_bak
        # check node prev_v in sub-path
        for h in range(-1,s):
            if work.prev(v) == (work.tour)[(i+h) % len(work.tour)]:
                diff_bak = float('inf')
        if diff_fwd <= diff_bak:
            return diff_fwd, 'fwd'
        else:
            return diff_bak, 'bak'

    # check tabu-list (add edges)
    def check_tabu(work, s, u, v, oper):
        i = (work.pos)[u]
        head_p, tail_p = u, (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % tsp.num_node], (work.tour)[(i+s) % len(work.tour)]
        if oper == 'fwd' and (work.get_tabu(prev_p, next_p) or work.get_tabu(v, head_p) or work.get_tabu(tail_p, work.next(v))):
            return True
        elif oper == 'bak' and (work.get_tabu(prev_p, next_p) or work.get_tabu(work.prev(v), tail_p) or work.get_tabu(head_p, v)):
            return True
        else:
            return False

    # update tabu-list (delete edges)
    def update_tabu(tsp, work, s, u, v, oper):
        i = (work.pos)[u]
        head_p, tail_p = u, (work.tour)[(i+s-1) % len(work.tour)]
        prev_p, next_p = (work.tour)[(i-1) % tsp.num_node], (work.tour)[(i+s) % len(work.tour)]
        work.set_tabu(prev_p,head_p)
        if oper == 'fwd':
            work.set_tabu(tail_p,next_p)
            work.set_tabu(v,work.next(v))
        else:
            work.set_tabu(tail_p,next_p)
            work.set_tabu(work.prev(v),v)

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

    # Or-opt neighborhood search
    min_delta, arg_min_delta, oper_min_delta = float('inf'), None, None
    nbhd = ((s,u,v)
            for s in range(1,size+1)
            for u in cur_work.tour
            for v in (tsp.neighbor)[u])
    for s,u,v in nbhd:
        asp_flag = False
        # evaluate difference
        delta, oper = eval_diff(tsp, cur_work, s, u, v)
        # update incumbnet solution
        if cur_work.obj + delta < work.obj:
            asp_flag = True
            work.copy(cur_work)
            change_tour(tsp, work, s, u, v, oper)
        # update best solution in the neighborhood
        if delta < min_delta and (not check_tabu(cur_work,s,u,v,oper) or asp_flag):
            min_delta = delta
            arg_min_delta = (s,u,v)
            oper_min_delta = oper
    if arg_min_delta is not None:
        # update tabu-list
        update_tabu(tsp, cur_work, *arg_min_delta, oper_min_delta)
        # update current working data
        change_tour(tsp, cur_work, *arg_min_delta, oper_min_delta)
    cur_work.cnt += 1
    if min_delta < 0:
        return True
    else:
        return False


# --------------------------------------------------------------------
#   3-opt neighborhood search
#
#   tsp(I): TSP data
#   work(I/O): working data
#   cur_work(I/O): current working data
#   return: [True] improved
# --------------------------------------------------------------------
def three_opt_search(tsp, work, cur_work):
    # evaluate difference for 3-opt operation
    def eval_diff_type134(tsp, work, u, v, w):
        diff, oper = float('inf'), None
        # evaluate type1
        cur_oper1 = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(w,work.next(w))
        new_oper1 = tsp.dist(u,v) + tsp.dist(work.prev(v),work.next(w)) + tsp.dist(w,work.next(u))
        if new_oper1 - cur_oper1 < diff and (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+1:
            diff, oper = new_oper1 - cur_oper1, 'type1'
        # evaluate type3
        cur_oper3 = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(work.prev(w),w)
        new_oper3 = tsp.dist(u,v) + tsp.dist(work.prev(w),work.prev(v)) + tsp.dist(work.next(u),w)
        if new_oper3 - cur_oper3 < diff and (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+2:
            diff, oper = new_oper3 - cur_oper3, 'type3'
        # evaluate type4
        cur_oper4 = tsp.dist(u,work.next(u)) + tsp.dist(v,work.next(v)) + tsp.dist(w,work.next(w))
        new_oper4 = tsp.dist(u,v) + tsp.dist(work.next(u),w) + tsp.dist(work.next(v),work.next(w))
        if new_oper4 - cur_oper4 < diff and (work.pos)[v] >= (work.pos)[u]+2 and (work.pos)[w] >= (work.pos)[v]+2:
            diff, oper = new_oper4 - cur_oper4, 'type4'
        return diff, oper

    def eval_diff_type2(tsp, work, u, v, w):
        cur_oper2 = tsp.dist(u,work.next(u)) + tsp.dist(work.prev(v),v) + tsp.dist(w,work.next(w))
        new_oper2 = tsp.dist(u,w) + tsp.dist(v,work.next(u)) + tsp.dist(work.prev(v),work.next(w))
        if (work.pos)[v] >= (work.pos)[u]+3 and (work.pos)[w] >= (work.pos)[v]+1:
            return new_oper2 - cur_oper2, 'type2'
        else:
            return float('inf'), None

    # check tabu list (add edges)
    def check_tabu(work, u, v, w, oper):
        if oper == 'type1' and (work.get_tabu(u,v) or work.get_tabu(work.prev(v),work.next(w)) or work.get_tabu(w,work.next(u))):
            return True
        elif oper == 'type2' and (work.get_tabu(u,w) or work.get_tabu(v,work.next(u)) or work.get_tabu(work.prev(v),work.next(w))):
            return True
        elif oper == 'type3' and (work.get_tabu(u,v) or work.get_tabu(work.prev(w),work.prev(v)) or work.get_tabu(work.next(u),w)):
            return True
        elif oper == 'type4' and (work.get_tabu(u,v) or work.get_tabu(work.next(u),w) or work.get_tabu(work.next(v),work.next(w))):
            return True
        else:
            return False

    # update tabu-list (delete edges)
    def update_tabu(work, u, v, w, oper):
        if oper == 'type1':
            work.set_tabu(u,work.next(u))
            work.set_tabu(work.prev(v),v)
            work.set_tabu(w,work.next(w))
        elif oper == 'type2':
            work.set_tabu(u,work.next(u))
            work.set_tabu(work.prev(v),v)
            work.set_tabu(w,work.next(w))
        elif oper == 'type3':
            work.set_tabu(u,work.next(u))
            work.set_tabu(work.prev(v),v)
            work.set_tabu(work.prev(w),w)
        elif oper == 'type4':
            work.set_tabu(u,work.next(u))
            work.set_tabu(v,work.next(v))
            work.set_tabu(w,work.next(w))

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

    # 3-opt neighborhood search
    min_delta, arg_min_delta, oper_min_delta = float('inf'), None, None
    # search type 1,3,4
    nbhd = ((u,v,w)
            for u in cur_work.tour
            for v in (tsp.neighbor)[u]
            for w in (tsp.neighbor)[cur_work.next(u)])
    for u,v,w in nbhd:
        asp_flag = False
        # evaluate difference
        delta, oper = eval_diff_type134(tsp, cur_work, u, v, w)
        # update incumbent solution
        if cur_work.obj + delta < work.obj:
            asp_flag = True
            work.copy(cur_work)
            change_tour(tsp, work, u, v, w, oper)
        # update best solution in the neighborhood
        if delta < min_delta and (not check_tabu(cur_work,u,v,w,oper) or asp_flag):
            min_delta = delta
            arg_min_delta = (u, v, w)
            oper_min_delta = oper
    # search type 2
    nbhd = ((u,v,w)
            for u in cur_work.tour
            for v in (tsp.neighbor)[cur_work.next(u)]
            for w in (tsp.neighbor)[u])
    for u,v,w in nbhd:
        asp_flag = False
        # evaluate difference
        delta, oper = eval_diff_type2(tsp, cur_work, u, v, w)
        # update incumbent solution
        if cur_work.obj + delta < work.obj:
            asp_flag = True
            work.copy(cur_work)
            change_tour(tsp, work, u, v, w, oper)
        # update best solution in the neighborhood
        if delta < min_delta and (not check_tabu(cur_work,u,v,w,oper) and asp_flag):
            min_delta = delta
            arg_min_delta = (u, v, w)
            oper_min_delta = oper
    if arg_min_delta is not None:
        # update tabu-list
        update_tabu(cur_work, *arg_min_delta, oper_min_delta)
        # update current working data
        change_tour(tsp, cur_work, *arg_min_delta, oper_min_delta)
    cur_work.cnt += 1
    if min_delta < 0:
        return True
    else:
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
    parser.add_argument('-t', '--time', help='time limit for tabu search', type=float, default=TIME_LIMIT)
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
    tabu_search(tsp, work, args.time)  # tabu search
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
