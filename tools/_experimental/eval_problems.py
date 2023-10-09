import json
from collections import namedtuple

perf_gt_path = "./perf_gt.json"


def format_print(task_id, task_res, base_or_plus):
    highest_offset = []
    lowest_offset = []
    avg_list = []
    for inp_task in task_res[base_or_plus]:
        inp = inp_task["input"]
        times = inp_task["rtime"]
        average = sum(times) / len(times)
        avg_list.append((str(inp), average))
        # offset
        highest_offset.append((max(times) - average) / average * 100)
        lowest_offset.append((average - min(times)) / average * 100)

    avg_list = sorted(avg_list, key=lambda x: x[1], reverse=True)

    print("{}\t{}\t{}%\t{}%\t".
          format(task_id, base_or_plus,
                 round(sum(highest_offset) / len(task_res[base_or_plus]), 2),
                 round(sum(lowest_offset) / len(task_res[base_or_plus]), 2)),
          end="")
    for inp, avg_time in avg_list:
        print("{}\t{}\t".format(inp, avg_time, 2), end="")
    print("")


if __name__ == "__main__":
    res = json.load(open(perf_gt_path, "r"))

    for task_id, task_res in res.items():
        format_print(task_id, task_res, "base")
        format_print(task_id, task_res, "plus")
