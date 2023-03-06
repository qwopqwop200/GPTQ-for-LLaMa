import argparse
import fnmatch
import tasks
import inspect
import functools


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='For OPT model to load; pass `facebook/opt-X`.\\ BLOOM model to load; pass `bigscience/bloom-X`'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=2, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--table_results', action="store_true", help='Print results in a table.'
    )

    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--wbits", type=int, default=32)
    parser.add_argument("--nearest", action="store_true")
    parser.add_argument('--load', type=str, default='')

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!

    return args
