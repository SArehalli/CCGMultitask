import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("input", type=str)
parser.add_argument("--output", type=str)

args = parser.parse_args()

with open(args.input, "rb") as input_f:
    w2idx = pickle.load(input_f)
    vocab = sorted(list(w2idx.keys()))

if args.output:
    with open(args.output, "w") as out_f:
        out_f.write("\n".join(vocab))
else:
    for word in vocab:
        print(word)
