import json
import jsonlines
from GPTFactory.factory import smart_build_factory
import os

DATA_ROOT = r"benchmark/MME"

with jsonlines.open(os.path.join(DATA_ROOT, "mme.jsonl")) as reader:
    data = [item for item in reader]

with open(r'api.json', 'r') as f:
    api_info = json.load(f)

'''
api_info = [{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10},{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10}], where tpm and rpm are the token_per_minute and request_per_minute limitation of each API
'''

worker_per_api = 4

factory = smart_build_factory(api_info,model='gpt-4-vision-preview',worker_num=worker_per_api*len(api_info),detail="high")


inputs = []

for idx,job in enumerate(data):
    imagename = job['image']
    img = os.path.join(DATA_ROOT, imagename)
    question = job['text']
    prompt = f"<img>{question}"
    img_dict = {"img":img}
    inputs.append({"prompt":prompt,"images":img_dict,"id":idx})

raw_results = factory.run_task(inputs,save_path=r'ckpts\mme',save_step=500,save_total_limit=2)

results = []

for item in raw_results:
    idx = item.id
    question_id = data[idx]['question_id']
    prompt = data[idx]['text']
    text = item.response
    results.append({"question_id":question_id, "prompt":prompt, "text":text})

with jsonlines.open(os.path.join(DATA_ROOT, "gpt4v-high_raw_answer.jsonl"), "w") as writer:
    writer.write_all(results)