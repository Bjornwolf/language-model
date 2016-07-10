import sys
import yaml
from cost_measurer import CostMeasurer

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
    ac_cost = float(line[-2])
    lm_cost = float(line[-1])
    choice_id = line[0].split('-')
    line = line[1:-2]
    line = reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', line))
    if choice_id[0] in phrases:
        phrases[choice_id[0]].append( (line, ac_cost, lm_cost) )
    else:
        phrases[choice_id[0]] = [(line, ac_cost, lm_cost)]
        
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

print len(phrases)
print len(correct_phrases)

oracle_per = {'i': 0., 'r': 0., 's': 0.}
total_length = 0.

for phrase_id in phrases.keys():
    correct_tokenised = cm.tokenise(correct_phrases[phrase_id])

    rank_list = [(p[0], dist(cm.tokenise(p[0]), correct_tokenised)) for p in phrases[phrase_id]]
    
    best_oracle, editions = min(rank_list, key=lambda x: x[1][0])
    best_oracle = cm.tokenise(best_oracle)

    distance, i, r, s = editions
    print 'NEXT:'
    print distance, i, r, s
    oracle_per['i'] += i
    oracle_per['r'] += r
    oracle_per['s'] += s

    total_length += len(correct_tokenised)
    
print 'Oracle PER: ', sum(oracle_per.values()) / total_length
print '  * insertions: ', oracle_per['i'] / total_length
print '  * deletions: ', oracle_per['r'] / total_length
print '  * substitutions: ', oracle_per['s'] / total_length

