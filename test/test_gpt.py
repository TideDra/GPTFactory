from GPTFactory import GPT
import os
YOUR_API_KEY = os.environ['OPENAI_API_KEY']
YOUR_END_POINT = os.environ['OPENAI_END_POINT']
chatbot = GPT(model='gpt-4-turbo',service='azure',api_key=YOUR_API_KEY,end_point=YOUR_END_POINT)
prompt = 'hello'
output = chatbot.complete(prompt)
print(output)