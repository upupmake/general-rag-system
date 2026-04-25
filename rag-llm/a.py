import json

import numpy as np


def calc_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def classify(x, y, test_x, history):
    pred = []
    for t_x in test_x:
        dists = [
            (i, calc_dist(cur_x, t_x))
            for i, cur_x in enumerate(x)
        ]
        dists.sort(key=lambda x: (x[1], x[0]))
        labels = [
            y[d[0]]
            for d in dists
        ]
        labels = labels[:3]
        # print(labels)
        acc = sum(labels)
        if acc == len(labels) / 2:
            label = labels[0]
        else:
            label = 1 if sum(labels) > len(labels) / 2 else 0
        pred.append(int(label))

    cur_m = np.array([len(x), len(x[0]), (len(y) - sum(y)) / len(x)])
    dists = [
        (i, calc_dist(cur_m, m["meta"]))
        for i, m in enumerate(history)
    ]
    dists.sort(key=lambda x: (x[1], x[0]))
    dists = dists[:3]
    scores = {}
    for d in dists:
        if "C" in history[d[0]]:
            k = history[d[0]]["C"]
            v = history[d[0]]["score"]
            if k not in scores:
                scores[k] = [v]
            else:
                scores[k].append(v)
    max_v = None
    target_k = None
    for k in scores:
        scores[k] = np.mean(scores[k])
        if max_v is None or scores[k] > max_v:
            max_v = scores[k]
            target_k = k
        elif scores[k] == max_v:
            if k < target_k:
                target_k = k
    return {"C_star": target_k, "pred": pred}


def main():
    lines = []
    while True:
        a = input()
        lines.append(a)
        if a == '}':
            break
    a = "".join(lines)
    a = json.loads(a)
    x = np.array(a["train_X"])
    y = np.array(a["train_y"])
    test_x = np.array(a["test_X"])
    history = np.array(a["history"])
    r = classify(x, y, test_x, history)
    print(json.dumps(r))


if __name__ == '__main__':
    main()
