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
gt_perf_times = 2


# execute code
def perf_exec(code, inputs, entry_point, output_not_none=False):
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


# perf framework
def perf_groundtruth(problems, hashcode, tasks_only_output_not_none, base_only=False):
    print("Perfiling Groundtruth...")
    tbegin = time.time()
    perf_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"] = perf_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        if base_only:
            perf_output[task_id] = oracle
            continue

        oracle["plus"] = perf_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        perf_output[task_id] = oracle
    print(f"Perfiled Groundtruth in {time.time() - tbegin:.2f}s")

    return perf_output


# select the most diffcult input within the CV threshold
def select_testcases(task_id, task_res, base_or_plus):
    OneInputResult = namedtuple("OneInputResult", ["inp", "avg", "CV", "outp"])
    testcases = []
    for inp_task in task_res[base_or_plus]:
        times = inp_task["rtime"]
        testcases.append(
            OneInputResult(
                inp=inp_task["input"],
                outp=inp_task["output"],
                avg=sum(times) / len(times) / 1e9,
                CV=np.std(times) / np.mean(times),
            )
        )

    filtered_testcases = [elm for elm in testcases if elm.CV < CV_threshold]

    result = {
        "task_id": task_id,
        "satis_input_num": len(filtered_testcases),
        "selected_input": [],
        "selected_output": [],
        "selected_CV": [],
        "selected_rtime": [],
    }
    if len(filtered_testcases):
        for testcase in filtered_testcases:
            result["selected_input"].append(testcase.inp)
            result["selected_output"].append(testcase.outp)
            result["selected_CV"].append(testcase.CV)
            result["selected_rtime"].append(testcase.avg)

    return result


# generate selected groundtruth
def get_selected_groundtruth(
    problems, hashcode, tasks_only_output_not_none, base_only=False
):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}_perf.pkl")
    if os.path.exists(cache_file):
        print(f"Load from perf groundtruth from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        perf_result = perf_groundtruth(
            problems, hashcode, tasks_only_output_not_none, base_only
        )

        expected_output = {}
        for task_id, task_res in perf_result.items():
            selected_results = select_testcases(task_id, task_res, "base")
            if not base_only:
                plus = select_testcases(task_id, task_res, "plus")
                selected_results["satis_input_num"] += plus["satis_input_num"]
                selected_results["selected_input"] += plus["selected_input"]
                selected_results["selected_output"] += plus["selected_output"]
                selected_results["selected_CV"] += plus["selected_CV"]
                selected_results["selected_rtime"] += plus["selected_rtime"]

            expected_output[task_id] = selected_results

        with open(cache_file, "wb") as f:
            pickle.dump(expected_output, f)

        return expected_output