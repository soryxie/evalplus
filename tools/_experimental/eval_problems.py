import argparse
import sys
import time
import os
import pickle
import numpy as np
from functools import reduce
from collections import namedtuple

from evalplus.data import (
    CACHE_DIR,
    get_human_eval_plus,
    get_human_eval_plus_hash,
)

CV_threshold = 0.02


# execute code
def perf_exec(code, inputs, entry_point, perf_time):
    exec_globals = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    reocords = []

    # Set perf_time as outside for-loop, in order to prevent cache
    for _ in range(perf_time):
        for index, inp in enumerate(inputs):
            start = time.time_ns()
            output = fn(*inp)
            latency = time.time_ns() - start

            if len(reocords) != len(inputs):
                reocords.append(
                    {"input": inp, "output": output, "rtime": [latency]})
            else:
                reocords[index]["rtime"].append(latency)

    return reocords


# perf framework
def perf_groundtruth(problems, hashcode, times):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}_perfGT.pkl")
    if os.path.exists(cache_file):
        print(f"Load cache from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Perfiling Groundtruth...")
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"] = perf_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            perf_time=times,
        )

        oracle["plus"] = perf_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            perf_time=times,
        )
        expected_output[task_id] = oracle
    print(f"Perfiled Groundtruth in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


# select the most diffcult input within the CV threshold
def select_input(task_id, task_res, base_or_plus):
    OneInputResult = namedtuple(
        "OneInputResult", ["inp", "avg", "CV", "outp"])
    testcases = []
    for inp_task in task_res[base_or_plus]:
        times = inp_task["rtime"]
        testcases.append(OneInputResult(
            inp=inp_task["input"],
            outp=inp_task["output"],
            avg=sum(times) / len(times),
            CV=np.std(times) / np.mean(times)
        ))

    problem_CV = reduce(lambda acc, ele: acc + ele.CV,
                        testcases, 0) / len(testcases)
#   lowest_CV = min(testcases, key=lambda x: x.CV)

    filtered_testcases = [elm for elm in testcases if elm.CV < CV_threshold]

    record = {
        "task_id": task_id,
        "problem_CV": problem_CV,
        "satis_input_num": len(filtered_testcases)}

    if len(filtered_testcases):
        max_testcase = max(filtered_testcases, key=lambda x: x.avg)
        record["selected_input"] = max_testcase.inp
        record["selected_output"] = max_testcase.outp
        record["selected_CV"] = max_testcase.CV
        record["selected_rtime"] = max_testcase.avg/1000000

    return record


# format print a table
def format_print(perf_res, output_file):
    with open(output_file, 'w') as f:
        sys.stdout = f
        for task_id, task_res in perf_res.items():
            for base_or_plus in ["base", "plus"]:
                record = select_input(task_id, task_res, base_or_plus)
                if record["satis_input_num"]:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        record["task_id"], base_or_plus, record["problem_CV"], record["satis_input_num"],
                        record["selected_input"], record["selected_CV"], record["selected_rtime"]))
                else:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                        record["task_id"], base_or_plus, record["problem_CV"], record["satis_input_num"], "", "", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", default=1, type=int)
    parser.add_argument("--res_file", default='./perf_results.txt', type=str)
    args = parser.parse_args()

    problems = get_human_eval_plus(mini=False)
    dataset_hash = get_human_eval_plus_hash()
    res = perf_groundtruth(problems, dataset_hash, args.times)

    format_print(res, args.res_file)


if __name__ == "__main__":
    main()
