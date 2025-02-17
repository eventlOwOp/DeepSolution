import os
import json
import tqdm
import time
import torch
import argparse
from utils.qwen_api import QwenAPI
from utils.embedder import Embedder
from ours_framework import GameTreeRAG
from utils.openai_api import OpenaiAPI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default="0_test")
    parser.add_argument('--worker_id', type=str, default="")
    parser.add_argument('--embedder_device', type=str, default="cpu")
    
    parser.add_argument('--model_name', type=str, default="qwen")
    parser.add_argument('--qwen_url', type=str, default="10.32.10.224")
    parser.add_argument('--qwen_url2', type=str, default="lzq")
    parser.add_argument('--qwen_url3', type=str, default="lzq")
    parser.add_argument('--qwen_url4', type=str, default="lzq")

    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--layer_top_k', type=int, default=1)
    parser.add_argument('--children_num', type=int, default=2)
    parser.add_argument('--retrieval_top_k', type=int, default=10)

    parser.add_argument('--doubt_max_new_tokens', type=int, default=2048)
    parser.add_argument('--solution_max_new_tokens', type=int, default=2048)
    parser.add_argument('--if_sum_reference', type=str, default="sum", choices=["sum", "notsum", "sumroot"])
    parser.add_argument('--if_only_reference', type=str, default="semionly", choices=["only", "notonly", "semionly"])
    parser.add_argument('--if_rerank', type=str, default="llmrerank", choices=["notrerank", "llmrerank"])
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--if_no_review', action='store_true')
    parser.add_argument('--if_no_explore', action='store_true')
    args = parser.parse_args() 
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    assert args.tag == "" or args.tag.startswith("_"), f"tag should be empty or start with _, i got {args.tag}"

    embedder = Embedder(device=args.embedder_device)
    if args.model_name == "qwen":
        llm = QwenAPI(
            url=f"http://{args.qwen_url}:1225/v1/chat/completions",
            url2=None if args.qwen_url2 is None else f"http://{args.qwen_url2}:1225/v1/chat/completions",
            url3=None if args.qwen_url3 is None else f"http://{args.qwen_url3}:1225/v1/chat/completions",
            url4=None if args.qwen_url4 is None else f"http://{args.qwen_url4}:1225/v1/chat/completions",
        )
        os.system(f"curl {args.qwen_url}:1225/v1/models --connect-timeout 2")
    else:
        raise NotImplementedError(f"model_name {args.model_name} not implemented")

    datas = json.load(open(f'./benchmark/{args.scenario}/datas.json'))
    if args.worker_id != "":
        datas = datas[int(args.worker_id)::8]
    print("len(datas)", len(datas), 'worker_id', args.worker_id)
    corpus = json.load(open(f'./benchmark/{args.scenario}/corpus.json'))    
    corpus_embeddings = torch.load(f'./benchmark/{args.scenario}/corpus.pt', map_location=embedder.model.device)
    print("corpus_embeddings.shape", corpus_embeddings.shape)

    framework = GameTreeRAG(
        embedder=embedder, 
        llm=llm,
        knowledge_lib=corpus, 
        knowledge_lib_embeddings=corpus_embeddings,
        max_depth=args.max_depth, 
        children_num=args.children_num, 
        layer_top_k=args.layer_top_k, 
        retrieval_top_k=args.retrieval_top_k, 
        doubt_max_new_tokens=args.doubt_max_new_tokens, 
        solution_max_new_tokens=args.solution_max_new_tokens, 
        if_only_reference=args.if_only_reference,
        if_sum_reference=args.if_sum_reference,
        if_rerank=args.if_rerank,
        if_no_review=args.if_no_review,
        if_no_explore=args.if_no_explore,
    )

    suffix = f"{args.max_depth}_{args.children_num}_{args.layer_top_k}{args.tag}"
    fw_dir = f"./eval_results/{args.scenario}"
    if not os.path.exists(fw_dir):
        os.makedirs(fw_dir)
    fw_path = f'{fw_dir}/generated_by_5_ours_{suffix}_{args.worker_id}.jsonl'
    fw = open(fw_path, 'a')
    exiting_data_ids = [data['id'] for data in [json.loads(line) for line in open(fw_path)]]
    for data in tqdm.tqdm(datas, desc=f"evaluating {suffix}, worker_id {args.worker_id}"):
        if data['id'] in exiting_data_ids:
            print(f"\n\nskip {data['id']}, already_done")
            continue
    
        print(f"\n\nwill deal with data {data['id']}")

        query = data['requirement']
        
        output_text, all_nodes_record = framework.get_final_solution(query)
        data['output_text'] = output_text
        data['all_nodes_record'] = all_nodes_record
        fw.write(json.dumps(data, ensure_ascii=False) + '\n')
        fw.flush()

    fw.close()
    print(f"evaluating 5_ours on {args.scenario} done, worker_id {args.worker_id}, results saved in {fw_path}")