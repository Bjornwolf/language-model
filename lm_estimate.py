import sys
import yaml
from cost_measurer import CostMeasurer
import numpy

COPY = 0
INSERTION = 1
DELETION = 2
SUBSTITUTION = 3

INFINITY = 10 ** 9


def _edit_distance_matrix(y, y_hat):
    """Returns the matrix of edit distances.

    Returns
    -------
    dist : numpy.ndarray
        dist[i, j] is the edit distance between the first
    action : numpy.ndarray
        action[i, j] is the action applied to y_hat[j - 1]  in a chain of
        optimal actions transducing y_hat[:j] into y[:i].
        i characters of y and the first j characters of y_hat.

    """
    dist = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
    action = dist.copy()
    for i in xrange(len(y) + 1):
        dist[i][0] = i
    for j in xrange(len(y_hat) + 1):
        dist[0][j] = j

    for i in xrange(1, len(y) + 1):
        for j in xrange(1, len(y_hat) + 1):
            if y[i - 1] != y_hat[j - 1]:
                cost = 1
            else:
                cost = 0
            insertion_dist = dist[i - 1][j] + 1
            deletion_dist = dist[i][j - 1] + 1
            substitution_dist = dist[i - 1][j - 1] + 1 if cost else INFINITY
            copy_dist = dist[i - 1][j - 1] if not cost else INFINITY
            best = min(insertion_dist, deletion_dist,
                       substitution_dist, copy_dist)

            dist[i][j] = best
            if best == insertion_dist:
                action[i][j] = action[i - 1][j]
            if best == deletion_dist:
                action[i][j] = DELETION
            if best == substitution_dist:
                action[i][j] = SUBSTITUTION
            if best == copy_dist:
                action[i][j] = COPY

    return dist, action


def edit_distance(y, y_hat):
    return _edit_distance_matrix(y, y_hat)[0][-1, -1]


def wer(y, y_hat):
    return edit_distance(y, y_hat) / float(len(y))



def dist(seq_1, seq_2):
    dists = [[(0, 0, 0, 0) for _ in range(len(seq_2) + 1)] for _ in range(len(seq_1) + 1)]
    for i in range(len(seq_1) + 1):
        dists[i][0] = (i, 0, i, 0)
    for i in range(len(seq_2) + 1):
        dists[0][i] = (i, i, 0, 0)
    for i in range(len(seq_1)):
        for j in range(len(seq_2)):
            cost = 1
            if seq_1[i] == seq_2[j]:
                cost = 0
            
            rem_cost = dists[i][j+1]
            ins_cost = dists[i+1][j]
            sub_cost = dists[i][j]
            if ins_cost[0] + 1 == min([ins_cost[0] + 1, rem_cost[0] + 1, sub_cost[0] + cost]):
                dists[i+1][j+1] = (ins_cost[0] + 1, ins_cost[1] + 1,
                                   ins_cost[2], ins_cost[3])
            elif rem_cost[0] + 1 == min([ins_cost[0] + 1, rem_cost[0] + 1, sub_cost[0] + cost]):
                dists[i+1][j+1] = (rem_cost[0] + 1, rem_cost[1],
                                   rem_cost[2] + 1, rem_cost[3])
            else:
                dists[i+1][j+1] = (sub_cost[0] + cost, sub_cost[1],
                                   sub_cost[2], sub_cost[3] + cost)

    return dists[-1][-1]



correct = sys.argv[1]
        
with open(correct) as f:
    correct_lines = f.readlines()

correct_phrases = {}
for line in correct_lines:
    line = line.split()
    choice_id = line[0]
    line = line[1:]
    line = reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', line))
    correct_phrases[choice_id] = line



# architectures = ['2x128', '2x256', '2x512', '3x128', '3x256', '3x512', '4x128', '4x256', '4x512']
architectures = ['3x512']

config_neural = 'configs/mgr/2x128.yaml'
cm = CostMeasurer(yaml.load(open(config_neural, 'r')))

beta = 4.
for name in architectures:
    beta = 4.
    while beta < 7.:
        baseline = sys.argv[2]

        neural_costs = {}
        for line in open(name + 'boot'):
            split_line = line.split()
            neural_costs[split_line[0]] = float(split_line[1])

        with open(baseline) as f:
            baseline_lines = f.readlines()
    
        phrases = {}
        for line in baseline_lines:
            line = line.split()
            ac_cost = float(line[-3])
            trans_cost = float(line[-1])
            lm_cost = neural_costs[line[0]]
            choice_id = line[0].split('-')
            line = line[1:-3]
            if line != []:
                line = reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', line))
            else:
                line = ''
            p_id = '-'.join(choice_id[:-1])
            if p_id in phrases:
                phrases[p_id].append( (line, ac_cost, trans_cost, lm_cost, choice_id[-1]) )
            else:
                phrases[p_id] = [(line, ac_cost, trans_cost, lm_cost, choice_id[-1])]

        total_phrases = 0
        neural_per = {'i': 0, 'r': 0, 's': 0}
        for phrase_id in phrases.keys():
            correct_tokenised = cm.tokenise(correct_phrases[phrase_id])
            rank_list = []
            for p in phrases[phrase_id]:
                rank_list.append((p[0], p[1], p[2], p[3]))
                
            best_neural = cm.tokenise(min(rank_list, key=lambda x: x[1] + 4 * x[2] + beta * (x[3]))[0])
            neural_distance, i, r, s = dist(correct_tokenised, best_neural)
            if edit_distance(correct_tokenised, best_neural) != neural_distance:
                print "!!!"
                print correct_tokenised
                print best_neural
            neural_per['i'] += i
            neural_per['r'] += r
            neural_per['s'] += s
        print name, beta, sum(neural_per.values()), neural_per['i'], neural_per['r'], neural_per['s']
        beta += 0.1
