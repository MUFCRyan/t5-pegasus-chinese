import json

import jsons

SCORE_FILE = './TrainLossScore.txt'


def find_max_score():
    with open(SCORE_FILE) as f:
        scores = f.read()
        scores = scores.replace('\n\t', '')
        scores = json.loads(scores)
        rougels = []
        for i in range(30):
            key = str(i)
            if key in scores.keys():
                rouge = scores[key]['rouge_l']
                rouge = float(rouge)
                rougels.append(rouge)

        max_score = max(rougels)
        print('max_score = {}'.format(max_score))


find_max_score()
