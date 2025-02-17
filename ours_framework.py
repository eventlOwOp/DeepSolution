import copy
import json 
import torch
import pickle
import random
random.seed(1225)
from utils.qwen_api import QwenAPI
from utils.embedder import Embedder
from utils.prompts import (
    prompt_for_get_doubt,
    prompt_for_get_doubt_proposal,
    prompt_for_get_solution,
    prompt_for_get_solution_proposal,
    prompt_for_get_solution_for_root,
    prompt_for_get_solution_proposal_for_root,  
)


class InNode: # initial node
    def __init__(self, question, my_id):
        self.question = question
        self.flag = "In"
        self.my_id = my_id
        self.children = []
        self.retrieval = []


class SoNode: # solution node
    def __init__(self, father, my_id):
        self.flag = "So"
        self.my_id = my_id
        self.proposal = ""
        self.proposal_input = ""
        self.solution = ""
        self.solution_input = ""
        self.solution_summary = ""
        self.retrieval = []
        self.retrieval_new = []
        self.father = father
        self.father_id = father.my_id
        self.score_for_father = None
        self.children = []
        self.scores_from_children = []
        self.if_final_used = False


class ReNode: # reflection node
    def __init__(self, father, my_id):
        self.flag = "Re"
        self.my_id = my_id
        self.proposal = ""
        self.proposal_input = ""
        self.reflection = ""
        self.reflection_input = ""
        self.reflection_summary = ""
        self.retrieval = []
        self.retrieval_new = []
        self.father = father
        self.father_id = father.my_id
        self.score_for_father = None
        self.children = []
        self.scores_from_children = []


class GameTreeRAG:
    def __init__(self, 
        knowledge_lib, 
        knowledge_lib_embeddings, 
        llm, 
        embedder, 
        max_depth, 
        children_num, 
        layer_top_k, 
        retrieval_top_k, 
        doubt_max_new_tokens, 
        solution_max_new_tokens,
        if_sum_reference, 
        if_only_reference, 
        if_rerank,
        if_no_review=False,
        if_no_explore=False
    ):
        self.node_id = 0
        self.root = None
        self.all_nodes_record = []
        self.all_nodes_record_dict = {}

        self.knowledge_lib = knowledge_lib
        self.knowledge_id_2_knowledge = {data['id']: data for data in knowledge_lib}
        self.knowledge_lib_embeddings = knowledge_lib_embeddings

        self.max_depth = max_depth
        self.children_num = children_num
        self.layer_top_k = layer_top_k 
        self.retrieval_top_k = retrieval_top_k

        self.llm = llm
        self.embedder = embedder

        self.doubt_max_new_tokens = doubt_max_new_tokens
        self.solution_max_new_tokens = solution_max_new_tokens

        self.if_only_reference = if_only_reference
        self.if_sum_reference = if_sum_reference
        self.if_rerank = if_rerank

        self.if_no_review = if_no_review
        self.if_no_explore = if_no_explore
        if if_no_review or if_no_explore:
            print("#################### ablation setting ####################")
            print(f"if_no_review: {if_no_review}, if_no_explore: {if_no_explore}")
            print("##########################################################")
            assert if_no_review == False or if_no_explore == False, "if_no_review and if_no_explore can not be True at the same time"
            if self.if_no_explore:
                self.layer_top_k = 1
                self.children_num = 1

    def get_final_solution(self, question, oracle_knowledge_ids=None):
        self.node_id = 0
        self.root = None
        self.all_nodes_record = []
        self.all_nodes_record_dict = {}

        self.root = InNode(question, my_id=self.node_id)
        self.all_nodes_record.append(self.root)
        self.node_id += 1

        good_solution_node = self.get_solution_tree()
        
        final_solution = good_solution_node.solution

        for node in self.all_nodes_record:
            if node.flag == "In":
                self.all_nodes_record_dict[str(node.my_id)] = {
                    "question": node.question,
                    "flag": node.flag,  
                    "my_id": node.my_id,
                    "retrieval": node.retrieval
                }
            elif node.flag == "So":
                self.all_nodes_record_dict[str(node.my_id)] = {
                    "proposal": node.proposal,
                    "proposal_input": node.proposal_input,
                    "solution": node.solution,
                    "solution_input": node.solution_input,
                    "solution_summary": node.solution_summary,
                    "retrieval": node.retrieval,
                    "retrieval_new": node.retrieval_new,
                    "flag": node.flag,
                    "my_id": node.my_id,
                    "father_id": node.father_id,
                    "score_for_father": node.score_for_father,
                    "scores_from_children": node.scores_from_children,
                    "if_final_used": node.if_final_used
                }
            elif node.flag == "Re":
                self.all_nodes_record_dict[str(node.my_id)] = {
                    "proposal": node.proposal,
                    "proposal_input": node.proposal_input,
                    "reflection": node.reflection,
                    "reflection_input": node.reflection_input,
                    "reflection_summary": node.reflection_summary,
                    "retrieval": node.retrieval,
                    "retrieval_new": node.retrieval_new,
                    "flag": node.flag,
                    "my_id": node.my_id,
                    "father_id": node.father_id,
                    "score_for_father": node.score_for_father,
                    "scores_from_children": node.scores_from_children,
                }
            else:
                raise ValueError("flag is not correct")

        return final_solution, self.all_nodes_record_dict

    def get_solution_tree(self):

        # 1. from root get four soultion children
        print_info = "current_depth: ROOT"
        print(f"\n\n======================================== {print_info}")
        self.root.retrieval = self.do_retrieval(query=self.root.question)
        self.root.children = self.get_children(self.root, children_num=self.layer_top_k*self.children_num, print_info=print_info)

        # 2. iter tree
        waiting_score_nodes = self.root.children # num is 4
        for current_depth in range(self.max_depth):

            new_children = [] # will be 4*2
            for n, node in enumerate(waiting_score_nodes):
                print_info = f"current_depth: {current_depth+1}/{self.max_depth}, node: {n+1}/{len(waiting_score_nodes)}, node.flag: {node.flag}"
                print(f"\n\n======================================== {print_info}")
                new_children += self.get_children(node, children_num=self.children_num, print_info=print_info)

            for node in new_children:
                node.father.children.append(node)
                node.father.scores_from_children.append(node.score_for_father)
            waiting_score_nodes_scores = [sum(node.scores_from_children) for node in waiting_score_nodes]
            top_k_indices = torch.argsort(torch.tensor(waiting_score_nodes_scores))[-self.layer_top_k:]
            scored_nodes = [waiting_score_nodes[i] for i in top_k_indices] # # num is 2

            waiting_score_nodes = []
            for node in scored_nodes:
                waiting_score_nodes += node.children

        assert all([node.flag == "Re" for node in waiting_score_nodes]), "not all are comment nodes"
        good_solution_node = waiting_score_nodes[0].father
        good_solution_node.if_final_used = True
        return good_solution_node

    def get_children(self, node, children_num=2, print_info=""):
        children = []
        if node.flag == "In": # will get solution nodes
            inputs = {"question": self.root.question}
            for _ in range(children_num):
                children.append(SoNode(father=node, my_id=self.node_id))
                self.node_id += 1
        elif node.flag == "So": # will get reflection nodes
            inputs = {"question": self.root.question, "solution": node.solution_summary}
            for _ in range(children_num):
                children.append(ReNode(father=node, my_id=self.node_id))
                self.node_id += 1
        elif node.flag == "Re": # will get solution nodes
            inputs = {"question": self.root.question, "solution": node.father.solution_summary, "reflection": node.reflection_summary}
            for _ in range(children_num):
                children.append(SoNode(father=node, my_id=self.node_id))
                self.node_id += 1
        else:
            raise ValueError("flag is not correct")
        self.all_nodes_record += children

        for c, child in enumerate(children):      
            print(f"\n=========do_proposal=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}")
            child.proposal_input, child.proposal = self.do_proposal(inputs, node.flag)

            print(f"\n=========do_retrieval=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}")
            child.retrieval_new = self.do_retrieval(child.proposal)
            child.retrieval = copy.deepcopy(child.retrieval_new)
            if self.if_sum_reference == "sum":
                new_retrieval_ids = [data['id'] for data in child.retrieval]
                for data in node.retrieval:
                    if data['id'] not in new_retrieval_ids:
                        child.retrieval.append(data)
                        new_retrieval_ids.append(data['id'])
                print(f"== sum reference, len(child.retrieval_new): {len(child.retrieval_new)}, len(child.retrieval): {len(child.retrieval)}")
            elif self.if_sum_reference == "sumroot":
                new_retrieval_ids = [data['id'] for data in child.retrieval]
                for data in self.root.retrieval:
                    if data['id'] not in new_retrieval_ids:
                        child.retrieval.append(data)
                        new_retrieval_ids.append(data['id'])
                print(f"== sum root reference, len(child.retrieval_new): {len(child.retrieval_new)}, len(child.retrieval): {len(child.retrieval)}")

            print(f"\n=========do_generation=========== {print_info}, child: {c+1}/{len(children)}, child.flag: {child.flag}")
            if node.flag == "Re" or node.flag == "In":
                child.score_for_father, child.solution_input, child.solution, child.solution_summary = self.do_generation(inputs, child.retrieval, node.flag)
            elif node.flag == "So":
                child.score_for_father, child.reflection_input, child.reflection, child.reflection_summary = self.do_generation(inputs, child.retrieval, node.flag)
            else:
                raise ValueError("flag is not correct")

        return children

    def do_proposal(self, inputs, father_flag):
        if father_flag == "In":
            print("## will get proposal based on father (In node)")
            input_text = prompt_for_get_solution_proposal_for_root.format(question=inputs['question'])
            proposal = self.llm.get_response(input_text, temperature=1.2, max_new_tokens=self.solution_max_new_tokens//2)
        elif father_flag == "So":
            if self.if_no_review: # ablation for no review, reflection is also solution
                print("## will get proposal based on father (So node), but reflection is also solution")
                input_text = prompt_for_get_solution_proposal.format(question=inputs['question'], solution=inputs['solution'], reflection=inputs['solution'])
                proposal = self.llm.get_response(input_text, temperature=1.2, max_new_tokens=self.solution_max_new_tokens//2)
                return input_text, proposal
            print("## will get proposal based on father (So node)")
            input_text = prompt_for_get_doubt_proposal.format(question=inputs['question'], solution=inputs['solution'])
            proposal = self.llm.get_response(input_text, temperature=1.2, max_new_tokens=self.doubt_max_new_tokens//2)
        elif father_flag == "Re":
            print("## will get proposal based on father (Re node)")
            input_text = prompt_for_get_solution_proposal.format(question=inputs['question'], solution=inputs['solution'], reflection=inputs['reflection'])
            proposal = self.llm.get_response(input_text, temperature=1.2, max_new_tokens=self.solution_max_new_tokens//2)
        else:
            raise ValueError("father_flag is not correct")
        return input_text, proposal

    def do_retrieval(self, query):
        query_embedding = self.embedder.get_embedding(query)
        scores = torch.nn.functional.cosine_similarity(query_embedding, self.knowledge_lib_embeddings, dim=1)
        top_k_datas = [self.knowledge_lib[i] for i in scores.argsort(descending=True)[:self.retrieval_top_k]]
        print(f"## retrieval knowledge, get {len(top_k_datas)} datas")
        return top_k_datas

    def do_generation(self, inputs, docs, father_flag):

        reference = ""
        for d, doc in enumerate(docs):
            reference += f"{d+1}. {doc['content']}\n"
        if self.if_only_reference == "semionly":
            if father_flag == "Re":
                print("## will add old solution and reflection to docs, because of semi-only reference")
                reference += f"{len(docs)+1}. {inputs['solution']}\n"
                reference += f"{len(docs)+2}. {inputs['reflection']}\n"
        reference = reference.strip()

        if self.if_rerank == "llmrerank":
            print("## will do llm rerank")
            input_text_for_rerank = f"<Instruction>:\nIn order to solve the following question, we retrieved a set of references. Some of these references are highly useful for addressing the problem, while others are less useful. Your task is to categorize these references into two groups: one group containing highly useful references and the other group containing less useful references.\n\n<Question>:\n{inputs['question']}\n\n<Reference>:\n{reference}\n\n<Output>:"
            rerank_output = self.llm.get_response(input_text_for_rerank, max_new_tokens=self.solution_max_new_tokens//2)
            reference = f"{reference}\n\n{rerank_output}"
            
        if father_flag == "In": # input: question, reference; output: solution
            print("## will get solution based on father (In node)")
            input_text = prompt_for_get_solution_for_root.format(question=inputs['question'], reference=reference)
            output_text = self.llm.get_response(input_text, max_new_tokens=self.solution_max_new_tokens)
            score = None
        elif father_flag == "So": # input: question, solution, reference; output: reflection
            if self.if_no_review: # ablation for no review, reflection is also solution
                print("## will get reflection based on father (So node), but reflection is also solution")
                if self.if_only_reference == "only" or self.if_only_reference == "semionly":
                    print("## will get solution based on father (Re node) by only reference or semi-only reference")
                    input_text = prompt_for_get_solution_for_root.format(question=inputs['question'], reference=reference)
                else:
                    print("## will get solution based on father (Re node) by not only reference, but also old solution and reflection")
                    input_text = prompt_for_get_solution.format(question=inputs['question'], solution=inputs['solution'], reflection=inputs['solution'], reference=reference)
                output_text = self.llm.get_response(input_text, max_new_tokens=self.solution_max_new_tokens)

                print("## will get score for father (So node), but reflection is also solution")
                input_text_for_score = f"The following is an old solution for the question, along with a doubt raised about the solution, and the new solution generated based on the doubt. You need to evaluate the effectiveness of the doubt, determining whether it effectively helped improve and refine the original solution.\n\n<Question>:\n{inputs['question']}\n\n<Solution>:\n{inputs['solution']}\n\n<Doubt>:\n{output_text}\n\n<New Solution>:\n{output_text}\n\n<Score>:\nThis doubt is effective."
                score = self.llm.get_prompt_prob(input_text_for_score, if_print=False)
                return score, input_text, output_text, output_text

            print("## will get reflection based on father (So node)")
            input_text = prompt_for_get_doubt.format(question=inputs['question'], solution=inputs['solution'], reference=reference)
            output_text = self.llm.get_response(input_text, max_new_tokens=self.doubt_max_new_tokens)

            print("## will get score for father (So node)")
            input_text_for_score = f"The following is a candidate solution for the question, along with a doubt raised about the solution. You need to evaluate the solution based on the doubt and assign it a score, with higher scores indicating better solutions.\n\n<Question>:\n{inputs['question']}\n\n<Solution>:\n{inputs['solution']}\n\n<Doubt>:\n{output_text}\n\n<Score>:\nThis solution is good."
            score = self.llm.get_prompt_prob(input_text_for_score, if_print=False)
        elif father_flag == "Re": # input: question, solution, reflection, reference; output: solution
            if self.if_only_reference == "only" or self.if_only_reference == "semionly":
                print("## will get solution based on father (Re node) by only reference or semi-only reference")
                input_text = prompt_for_get_solution_for_root.format(question=inputs['question'], reference=reference)
            else:
                print("## will get solution based on father (Re node) by not only reference, but also old solution and reflection")
                input_text = prompt_for_get_solution.format(question=inputs['question'], solution=inputs['solution'], reflection=inputs['reflection'], reference=reference)
            output_text = self.llm.get_response(input_text, max_new_tokens=self.solution_max_new_tokens)

            print("## will get score for father (Re node)")
            input_text_for_score = f"The following is an old solution for the question, along with a doubt raised about the solution, and the new solution generated based on the doubt. You need to evaluate the effectiveness of the doubt, determining whether it effectively helped improve and refine the original solution.\n\n<Question>:\n{inputs['question']}\n\n<Solution>:\n{inputs['solution']}\n\n<Doubt>:\n{inputs['reflection']}\n\n<New Solution>:\n{output_text}\n\n<Score>:\nThis doubt is effective."
            score = self.llm.get_prompt_prob(input_text_for_score, if_print=False)
        else:
            raise ValueError("father_flag is not correct")

        print("## will get summary for output_text")    
        output_text_summary_input = f"{output_text}\n\nGenerate a summary for the above text, the language should be English, and the summary should be concise and accurate."
        output_text_summary = self.llm.get_response(output_text_summary_input, max_new_tokens=self.solution_max_new_tokens//4)
        output_text_summary = output_text_summary.replace("\n", " ")

        return score, input_text, output_text, output_text_summary
