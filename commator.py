import sys
import yaml
from cost_measurer import CostMeasurer
import numpy


def get_best_comma(line, cm):
    sentence = line
    variants = [sentence]
    for i in range(1, len(sentence)):
        variants.append(line[:i] + [','] + line[i:])
    tokens = []
    for variant in variants:
        tokens.append(reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', variant)))
    best_comma = min(tokens, key=lambda x:cm.cost(x))
    while best_comma != tokens[0]:
        best_comma = best_comma.replace('><', ' ').replace('>', '').replace('<', '').split()
        variants = [best_comma]
        for i in range(1, len(best_comma)):
            variants.append(best_comma[:i] + [','] + best_comma[i:])
        tokens = []
        for variant in variants:
            tokens.append(reduce(lambda a, b: a + b, map(lambda x: '<' + x + '>', variant)))
        best_comma = min(tokens, key=lambda x:cm.cost(x))
    return best_comma
    # return best_comma.replace('><', ' ').replace('>', '').replace('<', ''), cm.cost(best_comma)


if __name__ == '__main__':
    config_neural = 'configs/mgr/3x512.yaml'
    cm = CostMeasurer(yaml.load(open(config_neural, 'r')))
    correct = sys.argv[1]
    with open(correct) as f:
        correct_lines = f.readlines()
    for line in correct_lines:
        line = line.split()
        choice_id = line[0]
        line = line[1:]
        best_comma = get_best_comma(line, cm)
        print best_comma
