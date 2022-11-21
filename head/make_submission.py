import argparse
import sys

import numpy as np

import data
from data import voting

parser = argparse.ArgumentParser()
parser.add_argument('--mean', default='geometric')
parser.add_argument('--out', default='submission.txt')
parser.add_argument('--out-probability', default='probability.npy')
parser.add_argument('files', nargs='+')
args = parser.parse_args()


def main():
    all_pps = []
    for path in args.files:
        pp, info = data.load(path)
        print(path, info, file=sys.stderr)
        all_pps.append(pp)

    if args.mean == 'vote':
        p = voting.vote(all_pps)
    else:
        pp = voting.mean(all_pps, args.mean)
        p = pp.argmax(axis=1)

    with open(args.out, 'w') as f:
        f.write("\n".join(str(int(round(x))) for x in p))

    if args.mean == 'vote':
        print('--mean=vote is specified. '
              'We cannot output prediction probability '
              'since aggregation by voting does not '
              'compute it. So, we skip it.')
    else:
        np.save(args.out_probability, pp)


if __name__ == '__main__':
    main()
