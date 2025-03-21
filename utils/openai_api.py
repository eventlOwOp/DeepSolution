import os
import json
import asyncio
import hashlib
import sqlite3
from typing import Optional
from filelock import FileLock
from openai import AsyncOpenAI


class OpenaiAPI:
    def __init__(
        self,
        cache_dir,
        api_key,
        model_name,
        base_url="https://yunwu.ai/v1",
    ):
        # 保存配置
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

        # 创建异步客户端
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # 设置缓存目录
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 缓存文件名
        self.cache_file_name = os.path.join(
            self.cache_dir, f"{model_name.replace('/', '_')}_cache.sqlite"
        )

    def _get_cache_key(self, messages, temperature=0.0, max_new_tokens=None):
        """生成缓存键"""
        key_data = {
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "model": self.model_name,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def _check_cache(self, key_hash):
        """检查缓存中是否存在结果"""
        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # 如果表不存在，创建它
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    metadata TEXT
                )
            """
            )
            conn.commit()

            c.execute("SELECT response, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()

            if row is not None:
                response, metadata_str = row
                metadata = json.loads(metadata_str)
                return response, metadata, True

            return None, None, False

    def _save_to_cache(self, key_hash, response, metadata):
        """保存结果到缓存"""
        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # 确保表存在
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    metadata TEXT
                )
            """
            )

            metadata_str = json.dumps(metadata)
            c.execute(
                "INSERT OR REPLACE INTO cache (key, response, metadata) VALUES (?, ?, ?)",
                (key_hash, response, metadata_str),
            )
            conn.commit()
            conn.close()

    async def get_response_async(self, prompt, temperature=0.7, max_new_tokens=None):
        """异步获取响应，使用缓存"""
        messages = [{"role": "user", "content": prompt}]

        # 生成缓存键
        key_hash = self._get_cache_key(messages, temperature, max_new_tokens)

        # 检查缓存
        cached_response, cached_metadata, cache_hit = self._check_cache(key_hash)
        if cache_hit:
            print(f"Async cache hit: {cached_response[:50]}...")
            print(f"Cache metadata: {cached_metadata}")
            return cached_response

        # 缓存未命中，使用异步API调用
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )

            resp = response.choices[0].message.content

            meta = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }

            print(f"Async API response: {resp[:50]}...")
            print(
                f"API metadata: prompt tokens: {meta['prompt_tokens']}; completion tokens: {meta['completion_tokens']}"
            )

            # 保存到缓存
            self._save_to_cache(key_hash, resp, meta)

            return resp
        except Exception as e:
            print(f"Async API call failed: {str(e)}")
            return f"Error: {str(e)}"

    async def get_prompt_prob_async(self, prompt):
        """异步获取提示的概率评分"""
        try:
            resp = await self.get_response_async(prompt)

            try:
                score = int(resp)
            except Exception as e:
                print(f"Error transforming response to int: {e}")
                score = 0

            print(f"Async score number: {score}")

            return score
        except Exception as e:
            print(f"Async API call failed: {str(e)}")
            return 0
