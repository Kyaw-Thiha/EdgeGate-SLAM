import argparse
from edgegate.viz.rerun_logger import log_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Path to persisted JSON training log")
    args = parser.parse_args()
    log_run(args.log_path)
