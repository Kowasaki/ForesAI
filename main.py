import json

from argparse import ArgumentParser
from benchmark import visualize_benchmark
from inference.detect import detect, old_detect

def main(args):
    with open(args.config_path, 'r') as j:
        config = json.load(j)
        # add cmd line args into config if they exist
        if args.res is not None:
            config["resolution"] = tuple(args.res)
        detect(config)
    if args.benchmark:
        visualize_benchmark.plot_resource_over_time(".")

if __name__ == "__main__":
    parser = ArgumentParser()
    # path of config file
    parser.add_argument("--config_path", required=True) 
    # Get benchmarks
    parser.add_argument("--benchmark", action='store_true', default=False) 
    # set resolution of videostream: width height
    # Only works if you are connecting to the stream directly (i.e. not via ROS)
    parser.add_argument("--res", nargs='+', type=int, default = None) 
    main(parser.parse_args())