import json
from collections import namedtuple

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="gpt4")
    parser.add_argument("--pb_name", type=str, default="HumanEval/152")
    args = parser.parse_args()

    AnalyseRes = namedtuple(
        'AnalyseRes', ['model', 'base_min', 'base_max', 'base_p90', 'plus_min', 'plus_max', 'plus_p90'])

    title_str = "Model\tbmin\tbmax\tbP90\tpmin\tpmax\tpP90"
    print(title_str)

    for model in args.models.split(","):
        res = json.load(open(model+".json", "r"))["eval"]

        evaldata = res.get(args.pb_name, None)

        if evaldata is None:
            print("Problem NOT found!")
        else:

            base_data = evaldata["base"][0][2]
            plus_data = evaldata["plus"][0][2]
            if base_data is None or len(base_data) == 0 or\
               plus_data is None or len(plus_data) == 0:
                record = AnalyseRes(model, 0, 0, 0, 0, 0, 0)
            else:
                record = AnalyseRes(
                    model=model,
                    base_min=min(base_data),
                    base_max=max(base_data),
                    base_p90=sorted(base_data)[int(len(base_data) * 0.9)],
                    plus_min=min(plus_data),
                    plus_max=max(plus_data),
                    plus_p90=sorted(plus_data)[int(len(plus_data) * 0.9)]
                )

            record_str = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                record.model, record.base_min, record.base_max, record.base_p90, record.plus_min, record.plus_max, record.plus_p90
            )
            print(record_str)
