import collections
import math
import jieba
import random
import dill
from logger import get_logger


class HmmSegmenter:
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']
        self.init_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}
        self.logger = get_logger("hmm")

    # to train hmm, we need to know 4 things:
    # 0. all possible state, for Chinese segment, all states are: "B", "M", "E", "S"
    # 1. init state prob
    # 2. trans prob matrix
    # 3. emit prob matrix
    def train_hmm(self, seqs):
        self.logger.info("training hmm, total input %d" % len(seqs))
        # default dict is used to set default value for map
        init_count_map = collections.defaultdict(lambda: 1)
        trans_count_map, emit_count_map = [collections.defaultdict(lambda: collections.defaultdict(lambda: 1))
                                           for _ in range(2)]
        for idx in range(len(seqs)):
            seq = seqs[idx]
            if idx != 0 and ((idx + 1) % 100 == 0 or idx == len(seqs) - 1):
                self.logger.info("training progress %d/%d" % (idx + 1, len(seqs)))
            observe = ''.join(seq)
            state = ''
            # seq is segmented, here we get the state sequence for the seq
            for term in seq:
                if len(term) == 1:
                    state += 'S'
                else:
                    state += 'B' + 'M' * (len(term) - 2) + 'E'
            if len(observe) != len(state):
                # data not legal
                continue
            init_count_map[state[0]] += 1
            # trans_prob count, from one state to next
            for i in range(1, len(state)):
                trans_count_map[state[i - 1]][state[i]] += 1
            # to calculate the emit-matrix prob
            for i in range(len(state)):
                emit_count_map[state[i]][observe[i]] += 1
        cnt = sum([init_count_map[k] for k in self.states])

        # prob is count/total_count
        # and prob is in "math.log", to avoid too small number
        for state in self.states:
            # init prob
            self.init_prob[state] = math.log(init_count_map[state] * 1.0 / cnt)

            total_count_trans = sum([trans_count_map[state][_] for _ in self.states])
            total_count_emit = sum([v for k, v in emit_count_map[state].items()])

            self.trans_prob[state] = {}
            self.emit_prob[state] = collections.defaultdict(lambda: math.log(1.0 / total_count_emit))

            for next_state in self.states:
                # prob from one state to next state
                self.trans_prob[state][next_state] = \
                    math.log(trans_count_map[state][next_state] * 1.0 / total_count_trans)

            for k, v in emit_count_map[state].items():
                # prob from one state to each Chinese word
                self.emit_prob[state][k] = math.log(v * 1.0 / total_count_emit)

    def cut(self, seq):
        # to cut a sequence, we need to get the state sequence of the sequence
        # here we use a dynamic programming algorithm to get the state sequence
        n = len(seq)

        path = [{} for _ in range(n)]
        dp = [{} for _ in range(n)]

        for j in self.states:
            dp[0][j] = self.init_prob[j] + self.emit_prob[j][seq[0]]
            path[0][j] = j

        # iterate the seq
        for i in range(1, n):
            for j in self.states:
                # use plus instead of multiply, because in each prob matrix, the prob is in "log"
                possible = []
                for k in self.states:
                    possible.append((dp[i - 1][k] + self.trans_prob[k][j] + self.emit_prob[j][seq[i]],
                                     path[i - 1][k] + j))

                # set the the max prob
                dp[i][j], path[i][j] = max(possible)

        # get the "state seq"
        _, state_seq = max([(dp[n - 1][k], path[n - 1][k]) for k in self.states])
        result, now = [], ''

        # from "state seq" get the segment result
        for i in range(n):
            if state_seq[i] == 'S':
                if len(now) > 0:
                    result.append(now)
                    now = ''
                result.append(seq[i])
            if state_seq[i] == 'M':
                now += seq[i]
            if state_seq[i] == 'E':
                now += seq[i]
                result.append(now)
                now = ''
            if state_seq[i] == 'B':
                if len(now) > 0:
                    result.append(now)
                    now = ''
                now += seq[i]
        if len(now) > 0:
            result.append(now)
        return result


def precision(test_data, actual):
    # a simplified way to calculate difference
    # if token in actual appears in test_data, then think it's correctly segmented
    x, y = 0, 0
    for i in range(len(test_data)):
        for term in actual[i]:
            x += (1 if term in test_data[i] else 0)
            y += 1
    return x * 1.0 / y


def main():
    with open("data/people's daily.txt", encoding='utf8') as f:
        lines = f.readlines()
    # line is like 19980101-01-003-001/m  北京/ns  举行/v  新年/t  音乐会/n
    data = []
    for line in lines:
        # in data, is like: 北京 举行 新年 音乐会
        data.append([s.split('/')[0] for s in line.split()[1:]])

    random.shuffle(data)

    # split train and test data
    train_data = data[: -int(len(data) * 0.2)]
    test_data = data[-int(len(data) * 0.2):]

    segmenter = HmmSegmenter()
    segmenter.train_hmm(train_data)
    dill.dump(segmenter, open("segmenter.pkl", "wb"))
    # cut with hmm
    hmm_result = [segmenter.cut(''.join(_)) for _ in test_data]

    # cut with jieba
    jieba_result = [[v for v in jieba.cut(''.join(_))] for _ in test_data]
    p_hmm, p_jieba = precision(test_data, hmm_result), precision(test_data, jieba_result)
    print('HMM Precision:', p_hmm)
    print('Jieba Precision:', p_jieba)


def load_model_test():
    segmenter = dill.load(open("segmenter.pkl", "rb"))
    custom_test_seqs = ["今天这里的天气真好",
                        "美国和东亚的关系发生了微妙的变化",
                        "算法智能带来麻烦，大数据社会需要考虑算法治理"]
    # third one cna be regarded as a bad case, because of the training data doesn't contain any
    # data about "大数据" or so
    for seq in custom_test_seqs:
        tks = segmenter.cut(seq)
        print("HMM:" + "__".join(tks) + "\n")


if __name__ == '__main__':
    main()
    load_model_test()
