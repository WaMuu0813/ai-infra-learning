import argparse
import logging

# parser = argparse.ArgumentParser()

# parser.add_argument("--model", type=str, required=True)
# parser.add_argument("--batch-size", type=int, default=1)
# parser.add_argument("--device", type=str, default="cpu")

# args = parser.parse_args()
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )

# # 给当前 Python 模块获取一个 logger
# logger = logging.getLogger(__name__) 

# logger.info("model = %s", args.model)
# logger.info("batch_size = %d", args.batch_size)
# logger.info("device = %s", args.device)

# logger.debug("debug information")
# logger.warning("this is a warning")
# logger.error("this is an error")

# print("model =", args.model)
# print("batch_size =", args.batch_size)
# print("device =", args.device)

import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def setup_logging(log_level):
    level=getattr(logging,log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def run(args):
    logger = logging.getLogger(__name__)

    logger.info("Starting inference")
    logger.info("model = %s", args.model)
    logger.info("batch_size = %d", args.batch_size)
    logger.info("device = %s", args.device)

    logger.info("Inference finished")

    logger.warning("this is a warning log")


def main():
    args = parse_args()

    setup_logging(args.log_level)

    run(args)


if __name__ == "__main__":
    main()