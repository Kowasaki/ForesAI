import json

from argparse import ArgumentParser
from benchmark import visualize_benchmark
from inference.detect import detect

def main(args):
    with open(args.config_path, 'r') as j:
        config = json.load(j)
        detect(config)
    if args.benchmark:
        visualize_benchmark.plot_resource_over_time(".")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--benchmark", action='store_true', default=False)
    main(parser.parse_args())