import os
import json
import time
import requests
from transformers import AutoTokenizer


class OpenaiAPI():
    def __init__(self, model_name="gpt-4o-2024-11-20"):
        print("model_name", model_name)

        # print("loading tokenizer")
        if os.path.exists("/data4/lizhuoqun2021/hf_models/llama-2-7b"):
            self.tokenizer = AutoTokenizer.from_pretrained("/data4/lizhuoqun2021/hf_models/llama-2-7b")
        elif os.path.exists("/mnt/data/lizhuoqun/hf_models/gpt2"):
            self.tokenizer = AutoTokenizer.from_pretrained("/mnt/data/lizhuoqun/hf_models/gpt2")
        else:
            raise Exception("No model path found")

        self.model_name = model_name

    def get_response(self, prompt):

        token_nums = len(self.tokenizer(prompt)['input_ids'])
        print(f"input_text token_nums: {token_nums}")

        url = "http://47.88.8.18:8088/api/ask"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjIzNzgzNiIsInBhc3N3b3JkIjoiMjM3ODM2MTIzIiwiZXhwIjoyMDMxMzc2MjA0fQ.Lz6IKLMUTWWT5isamrYTmbAcGNFpAqt87YFF2bynP3w"
        }
        raw_info = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
        }
        for r in range(125):
            callback = requests.post(url, data=json.dumps(raw_info), headers=headers, timeout=512)
            result = callback.json()
            print("callback", callback)
            if "200" in str(callback):
                break
            else:
                print(f"result: {result}")
                sleep_time = 10 * ((r+1) ** 2)
                print(f"sleeping {sleep_time}s after trying {r+1} times")
                time.sleep(sleep_time)

        return result['data']['response']['choices'][0]['message']['content']
