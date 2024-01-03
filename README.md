# GPTFactory

## About

GPTFactory is an all-in-one pipeline that collects data from ChatGPT models. You can use it to easily make data from OpenAI ChatGPT models!
It currently support some awesome features:
 - 🚀 Collecting data with multi-threads and multi APIs without extra code.
 - 🚀 Smartly assign APIs to threads for workload balance.
 - 🚀 Support the new GPT4-Vision model for image input.
 - 🚀 Limitation Checker that controls your API usage under the limitation and maximize efficiency.
 - 🚀 Easily resume task from checkpoint if it is interupted.

## Release Log
 - v0.1.1: fix hyperparameter bugs
 - v0.1.0: Support input few-shot examples
 - v0.0.1: Release GPTFactory

## Installation
```bash
git clone https://github.com/TideDra/GPTFactory
pip install .
```

## Usage
### Single-turn chat with a text-only chatbot
```python
from GPTFactory import GPT
chatbot = GPT(model='gpt-3.5-turbo',service='oai',api_key=YOUR_API_KEY,end_point=YOUR_END_POINT)
prompt = 'hello'
output = chatbot.complete(prompt)
print(output.response)
```

### Single-turn chat with a multimodal chatbot
```python
from GPTFactory import GPT
from GPTFactory import LimitationChecker
checker = LimitationChecker(token_rate_limit=10000, request_rate_limit=10)
chatbot = GPT(model='gpt-4-vision-preview',service='azure',api_key=YOUR_API_KEY,end_point=YOUR_END_POINT,limitation_checker=checker)
prompt = 'describe this image <img1>, and describe this image too <img2>.'
images = {"img1":'./img1.jpg',"img2":'./img2.jpg'}
output = chatbot.complete(prompt,images)
print(output.response)
```

### Single-turn chat with a multimodal chatbot given few-shot examples
```python
from GPTFactory import GPT
from GPTFactory import LimitationChecker
checker = LimitationChecker(token_rate_limit=10000, request_rate_limit=10)
chatbot = GPT(model='gpt-4-vision-preview',service='azure',api_key=YOUR_API_KEY,end_point=YOUR_END_POINT,limitation_checker=checker)
prompt = [
    {
        "role": "user",
        "content": "<img1>describe this image."
    },
    {
        "role": "assistant",
        "content": "There is a dog in the image."
    },
    {
        "role": "user",
        "content": "<img2>describe this image too."
    },
]
images = {"img1":'./img1.jpg',"img2":'./img2.jpg'}
output = chatbot.complete(prompt,images)
print(output.response)
```

### Process a batch of inputs using a factory
You can process a batch of inputs using multi threads and APIs with one simple method `run_task` 
```python
import json
import jsonlines
from GPTFactory.factory import smart_build_factory
import os

DATA_ROOT = r"benchmark\MME"

with jsonlines.open(os.path.join(DATA_ROOT, "mme.jsonl")) as reader:
    data = [item for item in reader]

with open(r'api.json', 'r') as f:
    api_info = json.load(f)

'''
api_info = [{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10},{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10}], where tpm and rpm are the token_per_minute and request_per_minute limitation of each API
'''

worker_per_api = 2

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
```

### Control the factory manually
We also support a more flexible operation style for the factory.
```python
import json
from GPTFactory.factory import smart_build_factory
import os

with open(r'api.json', 'r') as f:
    api_info = json.load(f)

'''
api_info = [{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10},{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10}], where tpm and rpm are the token_per_minute and request_per_minute limitation of each API
'''

worker_per_api = 2

factory = smart_build_factory(api_info,model='gpt-3.5-turbo',worker_num=worker_per_api*len(api_info),detail="high")

factory.start()
factory.put('hello')
output = factory.get()
print(output.response)
factory.stop()
```