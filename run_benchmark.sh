model_path=/data/songrun-data/evalperf/model

python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/vicuna-13b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/santacoder_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/stablelm-7b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codet5p-2b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/gptneo-2b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/vicuna-7b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/chatgpt_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/code-llama-34b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/starcoder_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/gpt-4_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen2-3b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codet5p-6b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codet5p-16b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/incoder-6b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen-6b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen2-7b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/code-llama-13b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen2-16b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/gpt-j_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/code-llama-7b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen2-1b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/incoder-1b_temp_0.0
python3 evalplus/evaluate.py --dataset humaneval --samples $model_path/codegen-2b_temp_0.0

# Gather results
python3 evalperf/tools/fetch_eval_results.py \
    --dataset humaneval \
    --models-dir $model_path/

# Benchmark
python3 evalperf/tools/eval_results.py --dataset humaneval 
