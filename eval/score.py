from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider


class Scorer:
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                '''for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))'''
                total_scores["Bleu"] = score
            else:
                # print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        '''for key, value in total_scores.items():
            print('{}:{}'.format(key, value))'''
        return total_scores


def calc_scores(predicts, titles, is_mt5, is_ground_truth=False):
    real_predicts = {}
    for index, predict in enumerate(predicts):
        if is_mt5:
            real_title = ' '.join(predict)
        else:
            real_title = predict
        real_predicts[index] = [real_title]
    real_titles = {}
    for index, title in enumerate(titles):
        if is_mt5:
            real_title = ' '.join(title)
        else:
            real_title = title
        if is_ground_truth:
            real_titles[index] = real_title
        else:
            real_titles[index] = [real_title]
    scorer = Scorer(real_predicts, real_titles)
    total_scores = scorer.compute_scores()
    return total_scores
