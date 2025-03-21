import copy
import torch
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple

random.seed(1225)
from utils.prompts import *
from utils.merge_list import MergeList
from rag import RAGBase


class InNode:  # initial node
    def __init__(self, question, my_id):
        self.question = question
        self.flag = "In"
        self.my_id = my_id
        self.children = []
        self.retrieval = MergeList()


class SoNode:  # solution node
    def __init__(self, father, my_id):
        self.flag = "So"
        self.my_id = my_id
        self.proposal = ""
        self.proposal_input = ""
        self.solution = ""
        self.solution_input = ""
        self.solution_summary = ""
        self.retrieval = MergeList()
        self.father = father
        self.father_id = father.my_id
        self.score_for_father = None
        self.children = []
        self.scores_from_children = []
        self.if_final_used = False


class ReNode:  # reflection node
    def __init__(self, father, my_id):
        self.flag = "Re"
        self.my_id = my_id
        self.proposal = ""
        self.proposal_input = ""
        self.reflection = ""
        self.reflection_input = ""
        self.reflection_summary = ""
        self.retrieval = MergeList()
        self.father = father
        self.father_id = father.my_id
        self.score_for_father = None
        self.children = []
        self.scores_from_children = []


class GameTreeRAG:
    def __init__(
        self,
        llm,
        rag: RAGBase,
        max_depth,
        children_num,
        layer_top_k,
        retrieval_top_k,
        doubt_max_new_tokens,
        solution_max_new_tokens,
        if_sum_reference,
        if_only_reference,
        if_rerank,
    ):
        self.node_id = 0
        self.root = None
        self.all_nodes_record = []
        self.all_nodes_record_dict = {}

        self.max_depth = max_depth
        self.children_num = children_num
        self.layer_top_k = layer_top_k
        self.retrieval_top_k = retrieval_top_k

        self.llm = llm
        self.rag = rag

        self.doubt_max_new_tokens = doubt_max_new_tokens
        self.solution_max_new_tokens = solution_max_new_tokens

        self.if_only_reference = if_only_reference
        self.if_sum_reference = if_sum_reference
        self.if_rerank = if_rerank

        # 进度跟踪
        self.progress = 0.0
        self.total_steps = 0
        self.completed_steps = 0

    def get_final_solution(self, question, oracle_knowledge_ids=None):
        """同步版本的入口点，内部调用异步方法"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.get_final_solution_async(question, oracle_knowledge_ids)
        )

    async def get_final_solution_async(self, question, oracle_knowledge_ids=None):
        """异步版本的解决方案生成"""
        self.node_id = 0
        self.root = None
        self.all_nodes_record = []
        self.all_nodes_record_dict = {}

        # 重置进度
        self.progress = 0.0
        self.completed_steps = 0

        # 估算总步骤数
        self.total_steps = self._estimate_total_steps()

        self.root = InNode(question, my_id=self.node_id)
        self.all_nodes_record.append(self.root)
        self.node_id += 1

        good_solution_node = await self.get_solution_tree_async()

        final_solution = good_solution_node.solution

        for node in self.all_nodes_record:
            self.update_node_record(node)

        # 完成所有步骤
        self.progress = 1.0

        return final_solution, self.all_nodes_record_dict

    def _estimate_total_steps(self):
        """估算总步骤数，用于进度计算"""
        # 根节点的子节点数
        root_children = self.layer_top_k * self.children_num

        # 每层的节点数和操作
        total = root_children  # 根节点的子节点生成

        # 对于每一层
        nodes_in_layer = root_children
        for _ in range(self.max_depth):
            # 每个节点生成的子节点数
            new_children = nodes_in_layer * self.children_num
            total += new_children  # 子节点生成

            # 下一层的节点数 (只保留top_k个节点的子节点)
            nodes_in_layer = self.layer_top_k * self.children_num

        # 每个节点有proposal, retrieval, generation三个步骤
        return total * 3

    def update_node_record(self, node):
        """Update node record in all_nodes_record_dict"""
        if node.flag == "In":
            self.all_nodes_record_dict[str(node.my_id)] = {
                "question": node.question,
                "flag": node.flag,
                "my_id": node.my_id,
                "retrieval": node.retrieval,
                "children": [child.my_id for child in node.children],
            }
        elif node.flag == "So":
            self.all_nodes_record_dict[str(node.my_id)] = {
                "proposal": node.proposal,
                "proposal_input": node.proposal_input,
                "solution": node.solution,
                "solution_input": node.solution_input,
                "solution_summary": node.solution_summary,
                "retrieval": node.retrieval,
                "flag": node.flag,
                "my_id": node.my_id,
                "father_id": node.father_id,
                "score_for_father": node.score_for_father,
                "scores_from_children": node.scores_from_children,
                "if_final_used": node.if_final_used,
                "children": [child.my_id for child in node.children],
            }
        elif node.flag == "Re":
            self.all_nodes_record_dict[str(node.my_id)] = {
                "proposal": node.proposal,
                "proposal_input": node.proposal_input,
                "reflection": node.reflection,
                "reflection_input": node.reflection_input,
                "reflection_summary": node.reflection_summary,
                "retrieval": node.retrieval,
                "flag": node.flag,
                "my_id": node.my_id,
                "father_id": node.father_id,
                "score_for_father": node.score_for_father,
                "scores_from_children": node.scores_from_children,
                "children": [child.my_id for child in node.children],
            }

    async def get_solution_tree_async(self):
        """异步版本的解决方案树生成"""
        # 1. from root get four solution children
        print_info = "current_depth: ROOT"
        print(f"\n\n======================================== {print_info}")

        # 异步执行检索
        retrieval_result = await self.do_retrieval_async(query=self.root.question)
        self.root.retrieval = MergeList(retrieval_result)
        self.update_node_record(self.root)

        # 异步获取子节点
        self.root.children = await self.get_children_async(
            self.root,
            children_num=self.layer_top_k * self.children_num,
            print_info=print_info,
        )

        # 2. iter tree
        waiting_score_nodes = self.root.children  # num is 4
        for current_depth in range(self.max_depth):
            new_children = []  # will be 4*2

            # 并行处理每个节点的子节点生成
            tasks = []
            for n, node in enumerate(waiting_score_nodes):
                print_info = f"current_depth: {current_depth+1}/{self.max_depth}, node: {n+1}/{len(waiting_score_nodes)}, node.flag: {node.flag}"
                print(f"\n\n======================================== {print_info}")
                task = self.get_children_async(
                    node, children_num=self.children_num, print_info=print_info
                )
                tasks.append(task)

            # 等待所有任务完成
            children_results = await asyncio.gather(*tasks)
            for children in children_results:
                new_children.extend(children)

            for node in new_children:
                node.father.children.append(node)
                node.father.scores_from_children.append(node.score_for_father)

            # 更新父节点记录
            for node in waiting_score_nodes:
                self.update_node_record(node)

            waiting_score_nodes_scores = [
                sum(node.scores_from_children) for node in waiting_score_nodes
            ]
            top_k_indices = torch.argsort(torch.tensor(waiting_score_nodes_scores))[
                -self.layer_top_k :
            ]
            scored_nodes = [waiting_score_nodes[i] for i in top_k_indices]  # # num is 2

            waiting_score_nodes = []
            for node in scored_nodes:
                waiting_score_nodes.extend(node.children)

        assert all(
            [node.flag == "Re" for node in waiting_score_nodes]
        ), "not all are comment nodes"
        good_solution_node = waiting_score_nodes[0].father
        good_solution_node.if_final_used = True
        self.update_node_record(good_solution_node)
        return good_solution_node

    async def get_children_async(self, node, children_num=2, print_info=""):
        """异步版本的子节点生成"""
        children = []
        if node.flag == "In":  # will get solution nodes
            inputs = {"question": self.root.question}
            for _ in range(children_num):
                children.append(SoNode(father=node, my_id=self.node_id))
                self.node_id += 1
        elif node.flag == "So":  # will get reflection nodes
            inputs = {"question": self.root.question, "solution": node.solution_summary}
            for _ in range(children_num):
                children.append(ReNode(father=node, my_id=self.node_id))
                self.node_id += 1
        elif node.flag == "Re":  # will get solution nodes
            inputs = {
                "question": self.root.question,
                "solution": node.father.solution_summary,
                "reflection": node.reflection_summary,
            }
            for _ in range(children_num):
                children.append(SoNode(father=node, my_id=self.node_id))
                self.node_id += 1
        else:
            raise ValueError("flag is not correct")
        self.all_nodes_record.extend(children)

        self.update_node_record(node)  # Update father node record

        # 并行处理每个子节点
        tasks = []
        for c, child in enumerate(children):
            task = self.process_child_async(
                child, c, children, inputs, node.flag, print_info
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        return children

    async def process_child_async(
        self, child, c, children, inputs, father_flag, print_info
    ):
        """异步处理单个子节点的所有步骤"""
        # 1. Proposal
        print(
            f"\n=========do_proposal=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}"
        )
        child.proposal_input, child.proposal = await self.do_proposal_async(
            inputs, father_flag
        )
        self.update_node_record(child)
        self._update_progress()

        # 2. Retrieval
        print(
            f"\n=========do_retrieval=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}"
        )
        retrieval_result = await self.do_retrieval_async(child.proposal)
        child.retrieval = MergeList(retrieval_result)

        if self.if_sum_reference == "sum":
            child.retrieval.append(child.father.retrieval)
            print(f"== sum reference, len(child.retrieval): {len(child.retrieval)}")
        elif self.if_sum_reference == "sumroot":
            child.retrieval.append(self.root.retrieval)
            print(
                f"== sum root reference, len(child.retrieval): {len(child.retrieval)}"
            )

        self.update_node_record(child)
        self._update_progress()

        # 3. Generation
        print(
            f"\n=========do_generation=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}"
        )
        if father_flag == "Re" or father_flag == "In":
            (
                child.score_for_father,
                child.solution_input,
                child.solution,
                child.solution_summary,
            ) = await self.do_generation_async(inputs, child.retrieval, father_flag)
        elif father_flag == "So":
            (
                child.score_for_father,
                child.reflection_input,
                child.reflection,
                child.reflection_summary,
            ) = await self.do_generation_async(inputs, child.retrieval, father_flag)
        else:
            raise ValueError("father_flag is not correct")

        self.update_node_record(child)  # Update child record after each major step
        self._update_progress()

    def _update_progress(self):
        """更新进度"""
        self.completed_steps += 1
        self.progress = min(0.99, self.completed_steps / self.total_steps)

    async def do_proposal_async(self, inputs, father_flag):
        """异步版本的提案生成"""
        if father_flag == "In":
            print("## will get proposal based on father (In node)")
            input_text = prompt_for_get_solution_proposal_for_root.format(
                question=inputs["question"]
            )
            proposal = await self.llm.get_response_async(
                input_text,
                temperature=1.2,
                max_new_tokens=self.solution_max_new_tokens // 2,
            )
        elif father_flag == "So":
            print("## will get proposal based on father (So node)")
            input_text = prompt_for_get_doubt_proposal.format(
                question=inputs["question"], solution=inputs["solution"]
            )
            proposal = await self.llm.get_response_async(
                input_text,
                temperature=1.2,
                max_new_tokens=self.doubt_max_new_tokens // 2,
            )
        elif father_flag == "Re":
            print("## will get proposal based on father (Re node)")
            input_text = prompt_for_get_solution_proposal.format(
                question=inputs["question"],
                solution=inputs["solution"],
                reflection=inputs["reflection"],
            )
            proposal = await self.llm.get_response_async(
                input_text,
                temperature=1.2,
                max_new_tokens=self.solution_max_new_tokens // 2,
            )
        else:
            raise ValueError("father_flag is not correct")
        return input_text, proposal

    async def do_retrieval_async(self, query):
        """异步版本的检索"""
        # 使用事件循环的run_in_executor来异步执行同步的检索方法
        loop = asyncio.get_event_loop()
        top_k_data = await loop.run_in_executor(
            None, self.rag.retrieve_top_k, query, self.retrieval_top_k
        )
        print(f"## retrieval knowledge, get {len(top_k_data)} data")
        return top_k_data

    async def do_generation_async(self, inputs, docs: MergeList, father_flag):
        """异步版本的生成"""
        reference = ""
        for d, doc in enumerate(docs):
            reference += f"{d+1}. {doc}\n"
        if self.if_only_reference == "semionly":
            if father_flag == "Re":
                print(
                    "## will add old solution and reflection to docs, because of semi-only reference"
                )
                reference += f"{len(docs)+1}. {inputs['solution']}\n"
                reference += f"{len(docs)+2}. {inputs['reflection']}\n"
        reference = reference.strip()

        if self.if_rerank == "llmrerank":
            print("## will do llm rerank")
            input_text_for_rerank = f"<Instruction>:\nIn order to solve the following question, we retrieved a set of references. Some of these references are highly useful for addressing the problem, while others are less useful. Your task is to categorize these references into two groups: one group containing highly useful references and the other group containing less useful references.\n\n<Question>:\n{inputs['question']}\n\n<Reference>:\n{reference}\n\n<Output>:"
            rerank_output = await self.llm.get_response_async(
                input_text_for_rerank, max_new_tokens=self.solution_max_new_tokens // 2
            )
            reference = f"{reference}\n\n{rerank_output}"

        if father_flag == "In":  # input: question, reference; output: solution
            print("## will get solution based on father (In node)")
            input_text = prompt_for_get_solution_for_root.format(
                question=inputs["question"], reference=reference
            )
            output_text = await self.llm.get_response_async(
                input_text, max_new_tokens=self.solution_max_new_tokens
            )
            score = None
        elif (
            father_flag == "So"
        ):  # input: question, solution, reference; output: reflection

            print("## will get reflection based on father (So node)")
            input_text = prompt_for_get_doubt.format(
                question=inputs["question"],
                solution=inputs["solution"],
                reference=reference,
            )
            output_text = await self.llm.get_response_async(
                input_text, max_new_tokens=self.doubt_max_new_tokens
            )

            print("## will get score for father (So node)")
            input_text_for_score = prompt_for_sol_eval.format(
                **inputs, doubt=output_text
            )
            score = await self.llm.get_prompt_prob_async(input_text_for_score)
        elif (
            father_flag == "Re"
        ):  # input: question, solution, reflection, reference; output: solution
            if self.if_only_reference == "only" or self.if_only_reference == "semionly":
                print(
                    "## will get solution based on father (Re node) by only reference or semi-only reference"
                )
                input_text = prompt_for_get_solution_for_root.format(
                    question=inputs["question"], reference=reference
                )
            else:
                print(
                    "## will get solution based on father (Re node) by not only reference, but also old solution and reflection"
                )
                input_text = prompt_for_get_solution.format(
                    question=inputs["question"],
                    solution=inputs["solution"],
                    reflection=inputs["reflection"],
                    reference=reference,
                )
            output_text = await self.llm.get_response_async(
                input_text, max_new_tokens=self.solution_max_new_tokens
            )

            print("## will get score for father (Re node)")
            input_text_for_score = prompt_for_rev_eval.format(
                **inputs, new_solution=output_text
            )
            score = await self.llm.get_prompt_prob_async(input_text_for_score)
        else:
            raise ValueError("father_flag is not correct")

        print("## will get summary for output_text")
        output_text_summary_input = f"{output_text}\n\nGenerate a summary for the above text, the language should be English, and the summary should be concise and accurate."
        output_text_summary = await self.llm.get_response_async(
            output_text_summary_input, max_new_tokens=self.solution_max_new_tokens // 4
        )
        output_text_summary = output_text_summary.replace("\n", " ")

        return score, input_text, output_text, output_text_summary
