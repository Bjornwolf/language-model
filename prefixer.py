import sys
import yaml
from cost_measurer import CostMeasurer
import numpy
import pickle
import random

config_neural = 'configs/mgr/3x512.yaml'
cm = CostMeasurer(yaml.load(open(config_neural, 'r')))

correct = sys.argv[1]
        
with open(correct) as f:
    correct_lines = f.readlines()

plots = []
correct_lines = filter(lambda y: len(y) == 57, map(lambda x: x.split()[1:], correct_lines))

print len(correct_lines)

for line in correct_lines:
    print len(plots)
    one_plot = []
    if len(line) == 57:
        for base in range(0, len(line)):
            xs = range(base, len(line))
            line_versions = [line[base:j+1] for j in xs]
            line_versions = map(lambda x: ''.join(map(lambda y: '<' + y + '>', x)), line_versions)
            costs = [0.] + map(lambda x: cm.cost(x), line_versions)
            ys = [costs[i] - costs[i-1] for i in range(1, len(costs))]
            one_plot.append( (line, xs, ys) )
        plots.append(one_plot)
pickle.dump(plots, open('prefix_plots_57', 'wb'))
