import argparse
from scripts.train import train_main
from scripts.evaluate import eval_main
from scripts.generate import generate_main
from preprocess import preprocess_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=["preprocess", "train", "evaluate", "generate"])
    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_main()
    elif args.command == "train":
        train_main()
    elif args.command == "evaluate":
        eval_main()
    elif args.command == "generate":
        generate_main()
