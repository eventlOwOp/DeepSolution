import time
import json
import requests
import os
import random
from transformers import AutoTokenizer


class QwenAPI:
    def __init__(self, url, url2="", url3="", url4=""):
        self.url = url
        self.url2 = url2
        self.url3 = url3
        self.url4 = url4
        self.url_list = [self.url, self.url2, self.url3, self.url4]
        self.url_list = [u for u in self.url_list if "." in u]
        print("url_list", self.url_list)
        if len(self.url_list) > 4:
            random_seed = len(str(os.urandom(12))) * len(str(os.urandom(25)))
            print("random_seed", random_seed)
            random.seed(random_seed)
            for i in range(len(self.url_list)):
                random_index = random.randint(0, len(self.url_list) - 1)
                print("random_index", random_index)
                self.url_list[i], self.url_list[random_index] = (
                    self.url_list[random_index],
                    self.url_list[i],
                )
            print("url_list after random", self.url_list)

        print("loading tokenizer")
        if os.path.exists("/data4/lizhuoqun2021/hf_models/llama-2-7b"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/data4/lizhuoqun2021/hf_models/llama-2-7b"
            )
        elif os.path.exists("/mnt/data/lizhuoqun/hf_models/gpt2"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                "/mnt/data/lizhuoqun/hf_models/gpt2"
            )
        else:
            raise Exception("No model path found")
        print("loading tokenizer done")

    def get_response(
        self,
        input_text,
        stop_str_list=[],
        temperature=0.7,
        max_new_tokens=4096,
        truction_thred=128000,
        truction_side="right",
        return_logprobs=False,
        if_print=True,
    ):
        current_time = time.time()

        input_text_token_num = len(self.tokenizer(input_text)["input_ids"])
        print(f"input_text_token_num: {input_text_token_num}")
        if input_text_token_num > truction_thred:
            print(
                f"we reduce the input_text_token_num to truction_thred {truction_thred}",
                "truction_side:",
                truction_side,
            )
            if truction_side == "right":
                input_text = input_text[
                    : int(len(input_text) * (truction_thred / input_text_token_num))
                ]
            elif truction_side == "left":
                input_text = input_text[
                    -int(len(input_text) * (truction_thred / input_text_token_num)) :
                ]
            else:
                raise Exception("truction_side not valid")
        url = self.url_select()
        headers = {
            # "Content-Type": "application/json",
            "Authorization": "EMPTY"
        }
        raw_info = {
            "model": "Qwen",
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": max_new_tokens,
            "stop": stop_str_list,
            "temperature": temperature,
            "logprobs": True,
            "echo": True,
        }

        data = json.dumps(raw_info)
        # print(data)

        for _ in range(2):
            try:
                callback = requests.post(url, headers=headers, data=data, timeout=10000)
                break
            except Exception as e:
                print("e", e)
                print("retry")
                url = self.url_select()

        if if_print:
            print("callback.status_code", callback.status_code)
        if callback.status_code != 200:
            print("callback.text", callback.text)
            raise Exception("callback.status_code != 200")
        # print("callback.json()", callback.json())

        if if_print:
            print(
                f"prompt_tokens: {callback.json()['usage']['prompt_tokens']}, total_tokens: {callback.json()['usage']['total_tokens']}, completion_tokens: {callback.json()['usage']['completion_tokens']}"
            )

        result = callback.json()
        # print(result)
        # print(result.keys())
        response = result["choices"][0]["message"]["content"]
        # print(response)
        # input()

        if if_print:
            print(
                "used time in this qwenapi get_response:",
                (time.time() - current_time) / 60,
                "min",
            )

        if return_logprobs is False:
            return response
        else:
            raw_probs = result["choices"][0]["logprobs"]["content"]
            avg_logprob = sum([raw_prob["logprob"] for raw_prob in raw_probs]) / len(
                raw_probs
            )
        return response, avg_logprob

    def get_prompt_prob(
        self, input_text, truction_thred=128000, truction_side="left", if_print=True
    ):
        current_time = time.time()

        input_text_token_num = len(self.tokenizer(input_text)["input_ids"])

        if if_print:
            print(f"input_text_token_num: {input_text_token_num}")

        if input_text_token_num > truction_thred:
            print(
                f"we reduce the input_text_token_num to truction_thred {truction_thred}",
                "truction_side:",
                truction_side,
            )
            if truction_side == "right":
                input_text = input_text[
                    : int(len(input_text) * (truction_thred / input_text_token_num))
                ]
            elif truction_side == "left":
                input_text = input_text[
                    -int(len(input_text) * (truction_thred / input_text_token_num)) :
                ]
            else:
                raise Exception("truction_side not valid")
        url = self.url_select()
        headers = {
            # "Content-Type": "application/json",
            "Authorization": "EMPTY"
        }
        raw_info = {
            "model": "Qwen",
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "echo": True,
        }

        data = json.dumps(raw_info)
        # print(data)

        for _ in range(2):
            try:
                callback = requests.post(
                    url, headers=headers, data=data, timeout=(10000, 10000)
                )
                break
            except Exception as e:
                print("e", e)
                print("retry")
                url = self.url_select()

        if if_print:
            print("callback.status_code", callback.status_code)
        if callback.status_code != 200:
            print("callback.text", callback.text)
            raise Exception("callback.status_code != 200")
        # print("callback.json()", callback.json())

        if if_print:
            print(
                f"prompt_tokens: {callback.json()['usage']['prompt_tokens']}, total_tokens: {callback.json()['usage']['total_tokens']}, completion_tokens: {callback.json()['usage']['completion_tokens']}"
            )

        result = callback.json()
        # print(result)
        # print(result.keys())
        # print(response)
        # input()

        raw_probs = result["prompt_logprobs"][-12:-5]
        avg_logprob = sum(
            [list(raw_prob.values())[0]["logprob"] for raw_prob in raw_probs]
        ) / len(raw_probs)

        if if_print:
            print(
                "used time in this qwenapi get_prompt_prob:",
                (time.time() - current_time) / 60,
                "min",
            )

        return avg_logprob

    def url_select(self):
        for url in self.url_list:
            curl_command = f"curl {url} --connect-timeout 1 -s"
            stream = os.popen(curl_command)
            output = stream.read()
            try:
                response_data = json.loads(output)
                break
            except json.JSONDecodeError as e:
                print(
                    f"!!!!!!!!!!!!!!!! {url} is not available !!!!!!!!!!!!!!!!, change to next url"
                )

        return url
