import sys
import yaml
from cost_measurer import CostMeasurer
import pickle

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

print dist('granit', 'granat')
print dist('grant', 'granat')
print dist('granat', 'granat')
print dist('pizza', 'kot')
print dist('p', 'kotkotkot')
print dist([24, 18,         38, 14, 23, 28],
           [24, 18, 19, 37, 38, 14, 23, 28])


correct = sys.argv[1]
baseline = sys.argv[2]
config_neural = sys.argv[3]

print correct
print baseline
print config_neural

with open(baseline) as f:
    baseline_lines = f.readlines()

phrases = {}
for line in baseline_lines:
    line = line.split()
    ac_cost = float(line[-3])
    trans_cost = float(line[-1])
    lm_cost = float(line[-2]) - trans_cost
    choice_id = line[0].split('-')
    line = line[1:-3]
    line = reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', line))
    if choice_id[0] in phrases:
        phrases[choice_id[0]].append( (line, ac_cost, trans_cost, lm_cost, choice_id[1]) )
    else:
        phrases[choice_id[0]] = [(line, ac_cost, trans_cost, lm_cost, choice_id[1])]
        
with open(correct) as f:
    correct_lines = f.readlines()

correct_phrases = {}
for line in correct_lines:
    line = line.split()
    choice_id = line[0]
    line = line[1:]
    line = reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', line))
    correct_phrases[choice_id] = line


cm = CostMeasurer(yaml.load(open(config_neural, 'r')))
print cm.numbers_from_text

better_neural = 0.
better_baseline = 0.
total_phrases = 0.
neural_per = {'i': 0., 'r': 0., 's': 0.}
baseline_per = {'i': 0., 'r': 0., 's': 0.}
total_length = 0.
out_neural = open(sys.argv[4], 'wb')
for phrase_id in phrases.keys():
    correct_tokenised = cm.tokenise(correct_phrases[phrase_id])
    bb = phrases[phrase_id][0][0]
    best_baseline = cm.tokenise(bb)
    baseline_distance, i, r, s = dist(correct_tokenised, best_baseline)
    '''
    print 'NEXT:'
    print len(correct_tokenised)
    print correct_tokenised
    print best_baseline
    print baseline_distance, i, r, s
    '''
    baseline_per['i'] += i
    baseline_per['r'] += r
    baseline_per['s'] += s

    rank_list = []
    for p in phrases[phrase_id]:
        out_line = phrase_id + '-' + p[4]
        cost = cm.cost(p[0])
        out_line += ' ' + str(cost) + '\n'
        rank_list.append((p[0], p[1], p[2], cost))
        out_neural.write(out_line)
        
    best_neural = cm.tokenise(min(rank_list, key=lambda x: x[1] + 4 * (x[2] + x[3]))[0])
    neural_distance, i, r, s = dist(correct_tokenised, best_neural)
    '''
    print best_neural
    print neural_distance, i, r, s
    '''
    neural_per['i'] += i
    neural_per['r'] += r
    neural_per['s'] += s

    if neural_distance < baseline_distance:
        better_neural += 1
    elif neural_distance > baseline_distance:
        better_baseline += 1
    total_phrases += 1
    total_length += len(correct_tokenised)
    
print 'Total phrases: ', total_phrases
print 'Neural advantage: ', better_neural
print 'Baseline advantage: ', better_baseline
print 'Neural PER: ', sum(neural_per.values()) # / total_length
print '  * insertions: ', neural_per['i'] # / total_length
print '  * deletions: ', neural_per['r'] # / total_length
print '  * substitutions: ', neural_per['s'] # / total_length
print 'Baseline PER: ', sum(baseline_per.values()) # / total_length
print '  * insertions: ', baseline_per['i'] # / total_length
print '  * deletions: ', baseline_per['r'] # / total_length
print '  * substitutions: ', baseline_per['s'] # / total_length
