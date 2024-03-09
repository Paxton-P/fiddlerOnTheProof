"""
You are the manager for the New York Frets, a baseball team that has woefully underperformed this season. In an effort
to right the ship, you are tinkering with the batting order. Eight of your nine batters are “pure contact” hitters.
One-third of the time, each of them gets a single, advancing any runners already on base by exactly one base. (Your team
is very slow on the base paths. That means no one is fast enough to score from first or second base on a single—only
from third). The other two-thirds of the time, they record an out, and no runners advance to the next base. Your ninth
batter is the slugger. One-tenth of the time, he hits a home run. But the remaining nine-tenths of the time, he strikes
out. Your goal is to score as many runs as possible, on average, in the first inning. Where in your lineup (first,
second, third, etc.) should you place your home run slugger?
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    expvs = np.zeros(9)
    for i in range(9):
        expvs[i] = sim_lineup(position=i)

    print(expvs)

def sim_lineup(position=-1):
    n = 10000000
    scores = np.zeros(n)
    for i in range(n):
        scores[i] = inning(position)

    unique, counts = np.unique(scores, return_counts=True)
    expv = np.sum([score * count for score, count in zip(unique, counts)]) / n
    print(expv)
    # plt.hist(scores, bins=7, range=(0, 7))
    # plt.show()

    return expv


def inning(position=-1):
    batters = 9

    outs = 0
    points = 0
    bases_filled = 0

    for i in range(batters):
        if i == position:
            hit = slug()
            if hit:
                points += bases_filled + 1
                bases_filled = 0
            else:
                outs += 1
        else:
            hit = bunt()
            if hit:
                if bases_filled == 3:
                    points += 1
                else:
                    bases_filled += 1
            else:
                outs += 1

        if outs == 3:
            return points
    return points


def bunt():
    r = random.uniform(0, 1)
    if r < 1/3:
        return True
    else:
        return False


def slug():
    r = random.uniform(0, 1)
    if r < 1 / 10:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
    exit(0)
