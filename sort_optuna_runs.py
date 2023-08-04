import argparse
import json


def readFile(file_name):
    trials = []
    with open(file_name, "r") as data:
        for line in data.readlines():
            trials.append(json.loads(line))
    return trials


def printBestScores(score_sorted, n_best):
    print(f"{n_best} best models based on score")
    for i, params in enumerate(score_sorted):
        if i == n_best:
            break
        [macs_value, macs_unit] = params["MACs"].split(" ")
        print("{0:3}. Score: {1:<18}   MACs: {2:>6.2f} {3:}".format(
            i+1, params['score'], float(macs_value), macs_unit))


def printBestMACs(macs_sorted, n_best):
    print(f"\n{n_best} best models based on MACs")
    for i, params in enumerate(macs_sorted):
        if i == n_best:
            break
        [macs_value, macs_unit] = params["MACs"].split(" ")
        print("{0:3}. MACs: {1:>6.2f} {2:}   Score: {3:<18}".format(
            i+1, float(macs_value), macs_unit, params['score']))


def printModelParams(params, i):
    if i is not None:
        print(params[i])


def main():
    # Read file name from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="Path to file")
    parser.add_argument("--score_i", "-si", required=False, type=int, help="Score index too print")
    parser.add_argument("--macs_i", "-mi", required=False, type=int, help="MACs index too print")
    parser.add_argument("--n_best", "-nb", required=False, type=int, default=10, help="Number of best models printed")
    args = parser.parse_args()

    # Read and sort
    trials = readFile(args.file)
    score_sorted = sorted(trials, key=lambda score: -score["score"])
    macs_sorted = sorted(trials, key=lambda macs: float(macs["MACs"].split(" ")[0]))

    printBestScores(score_sorted, args.n_best)
    printBestMACs(macs_sorted, args.n_best)
    printModelParams(score_sorted, args.score_i)
    printModelParams(macs_sorted, args.macs_i)


main()
