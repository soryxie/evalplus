import argparse
import os
import pickle
import sys
import time
from collections import namedtuple
from functools import reduce

import numpy as np

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.data.utils import CACHE_DIR

CV_threshold = 0.02
gt_perf_times = 5


def perf_exec(code, inputs, entry_point, output_not_none=False):
    """
    Perf with every input in inputs
    Output: [{"input": input, "output": output, "rtime": [runtimes]}]
    """
    exec_globals = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    reocords = []
    # Set perf_time as outside for-loop, in order to prevent cache
    for _ in range(gt_perf_times):
        for index, inp in enumerate(inputs):
            start = time.time_ns()
            output = fn(*inp)
            latency = time.time_ns() - start

            if len(reocords) != len(inputs):
                reocords.append({"input": inp, "output": output, "rtime": [latency]})
            else:
                reocords[index]["rtime"].append(latency)

    if output_not_none:
        for ret in reocords:
            ret["output"] = True if ret["output"] is not None else False

    return reocords


def perf_groundtruth(problems, hashcode, tasks_only_output_not_none):
    """
    Perf groundtruth for all tasks
    Output: {task_id: { "base": [{"input": input, "output": output, "rtime": [runtimes]}], 
                        "plus": [{"input": input, "output": output, "rtime": [runtimes]}]}}
    """
    print("Perfiling Groundtruth...")
    tbegin = time.time()
    perf_output = {}

    for task_id, problem in problems.items():
        records = perf_exec( # base
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        records += perf_exec( # plus
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        perf_output[task_id] = records

    print(f"Perfiled Groundtruth in {time.time() - tbegin:.2f}s")
    return perf_output


def select_testcases(task_id, task_results):
    """
    Select testcases for one task
    Perf every input 5 times, calculate 5 runtimes' CV, if CV < 0.02, select this input
    """
    OneInputRecord = namedtuple("OneInputRecord", ["input", "output", "runtime_avg", "runtime_cv"])
    testcases = []

    for result in task_results:
        times = result["rtime"]
        testcases.append(
            OneInputRecord(
                input       = result["input"],
                output      = result["output"],
                runtime_avg = sum(times) / len(times) * 1e-9,
                runtime_cv  = np.std(times) / np.mean(times),
            )
        )

    filtered_testcases = [elm for elm in testcases if elm.runtime_cv < CV_threshold]

    result = {
        "task_id": task_id, "satis_input_num": len(filtered_testcases),
        "selected_input": [], "selected_output": [], "selected_rtime": [],
    }
    for testcase in filtered_testcases:
        result["selected_input"].append(testcase.input)
        result["selected_output"].append(testcase.output)
        result["selected_rtime"].append(testcase.runtime_avg)
    return result


def get_groundtruth_with_selected_testcases(problems, hashcode, tasks_only_output_not_none):
    """
    Get groundtruth with selected testcases
    Output: {task_id: { "task_id": task_id, "satis_input_num": int, 
                        "selected_input": [inputs], "selected_output": [outputs], "selected_rtime": [runtimes]}}
    """
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}_perf.pkl")
    if os.path.exists(cache_file):
        print(f"Load from perf groundtruth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        perf_result = perf_groundtruth(problems, hashcode, tasks_only_output_not_none)
        expected_output = {
            task_id: select_testcases(task_id, task_results) 
                for task_id, task_results in perf_result.items()
        }
        with open(cache_file, "wb") as f:
            pickle.dump(expected_output, f)
        return expected_output