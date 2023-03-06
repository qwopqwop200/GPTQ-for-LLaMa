from pprint import pprint
from typing import List, Union
from .tasks_utils import Task
from . import piqa
from . import arc
from . import superglue
from .local_datasets import lambada as lambada_dataset
from .lambada import LAMBADA
from . import glue
from . import storycloze

# TODO: Add the rest of the results!
########################################
# All tasks
########################################


TASK_REGISTRY = {
    "lambada": LAMBADA,
    "piqa": piqa.PiQA,
    "arc_easy": arc.ARCEasy,
    "arc_challenge": arc.ARCChallenge,
    "boolq": superglue.BoolQ,
    "cb": superglue.CommitmentBank,
    "copa": superglue.Copa,
    "wic": superglue.WordsInContext,
    "multirc": superglue.MultiRC,
    "rte": glue.RTE,
    "record": superglue.ReCoRD,
    "wsc": superglue.SGWinogradSchemaChallenge,
    "storycloze": storycloze.StoryCloze2018
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_dict(task_name_list: List[Union[str, Task]]):
    task_name_dict = {
        task_name: get_task(task_name)()
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if not isinstance(task_object, str)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {**task_name_dict, **task_name_from_object_dict}
