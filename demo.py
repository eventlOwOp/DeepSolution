import json, json5
import asyncio
from ours_framework import GameTreeRAG
from utils.openai_api import OpenaiAPI
from rag import HippoRAGClient


async def async_main():
    rag = HippoRAGClient(
        save_dir="../HippoRAG/outputs",
        llm_model_name="gpt-4o-mini",
        llm_base_url="https://xiaoai.plus/v1",
        embedding_model_name="text-embedding-3-large",
        llm_api_key="***",
    )

    # 创建LLM客户端
    llm = OpenaiAPI(
        cache_dir="./cache",
        base_url="https://xiaoai.plus/v1",
        api_key="***",
        model_name="gpt-4o-mini",
    )

    framework = GameTreeRAG(
        llm=llm,
        rag=rag,
        max_depth=3,
        children_num=2,
        layer_top_k=1,
        retrieval_top_k=10,
        doubt_max_new_tokens=2048,
        solution_max_new_tokens=2048,
        if_only_reference="semionly",  # only, notonly, semionly
        if_sum_reference="sum",  # sum, notsum, sumroot
        if_rerank="llmrerank",  # notrerank, llmrerank
    )

    # query = "讨论术中是否保留膀胱颈"
    # query = "RARP术后要注意什么"

    output_text, all_nodes_record = await framework.get_final_solution_async(query)

    print(output_text)
    with open("result.json5", "w", encoding="utf8") as f:
        json5.dump(all_nodes_record, f, ensure_ascii=False)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
