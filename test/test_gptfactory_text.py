import json
import jsonlines
from GPTFactory.factory import smart_build_factory
import os

with open(r'api_gpt4.json', 'r') as f:
    api_info = json.load(f)

'''
api_info = [{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10},{"api_key":"xxx","end_point":"xxx","tpm":800,"rpm":10}], where tpm and rpm are the token_per_minute and request_per_minute limitation of each API
'''

worker_per_api = 4

factory = smart_build_factory(api_info,model='gpt-4',worker_num=2*len(api_info),service='azure',rpm=400,tpm=4e4,debug=True)


inputs = []

for idx in range(100):

    inputs.append({"prompt":"Is 1+1 = 2 ?","id":idx})

raw_results = factory.run_task(inputs)