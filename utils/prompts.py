prompt_for_get_solution_proposal_for_root = """<Instruction>:
In order to solve the following question, I need to search for relevant knowledge in external knowledge bases. Please generate a proposal based on the question and tell me what areas of knowledge I should search for.

<Question>:
{question}

<Proposal>:

"""


prompt_for_get_doubt_proposal = """<Instruction>:
For the following question, there is currently a candidate solution. To evaluate this candidate solution, I need to search for relevant knowledge in external knowledge bases. Please generate a proposal based on the question and the candidate solution, and tell me what areas of knowledge I should search for.

<Question>:
{question}

<Candidate Solution>:
{solution}

<Proposal>:"""


prompt_for_get_solution_proposal = """<Instruction>:
For the following question, there is a candidate solution as well as an expert's evaluation of this candidate solution. In order to redesign a better solution, I need to search for relevant knowledge in external knowledge bases. Please generate a proposal based on the question and the candidate solution, and tell me what areas of knowledge I should search for.

<Question>:
{question}

<Candidate Solution>:
{solution}

<Critique for Candidate Solution>:
{reflection}

<Proposal>:"""


prompt_for_get_solution_for_root = """<Instruction>:
Based on the reference knowledge, design a good solution for the question. Be sure to make full use of reference knowledge to analyze the challenges contained within the question and provide a comprehensive solution.

<Question>:
{question}

<Reference>:
{reference}

<Solution>:"""


prompt_for_get_doubt = """<Instruction>: 
For the following question, a candidate solution has already been provided. You need to critique the candidate solution based on the reference knowledge. Be sure to make full use of the reference knowledge to identify the shortcomings of the old solution in terms of its analysis of the challenges in the question and its technical implementation.

<Question>:
{question}

<Candidate Solution>:
{solution}

<Reference>:
{reference}

<Critique>:"""


prompt_for_get_solution = """<Instruction>:
For the following question, an old solution has already been provided and its shortcomings have been pointed out by human experts. You need to redesign a better solution based on the reference knowledge and the guidance from human experts. Be sure to make full use of the reference knowledge to analyze the challenges contained within the question and provide a comprehensive solution.

<Question>:
{question}

<Candidate Solution>:
{solution}

<Critique for Candidate Solution>:
{reflection}

<Reference>:
{reference}

<New Solution>:"""


prompt_for_sol_eval = """<Instruction>:
The following is a candidate solution for the question, along with a doubt raised about the solution. You need to evaluate the solution based on the doubt and assign it a score, with higher scores indicating better solutions.
Respond with only a score number from 0 to 10

<Question>:
{question}

<Solution>:
{solution}

<Doubt>:
{doubt}

<Score>:"""


prompt_for_rev_eval = """<Instruction>:
The following is an old solution for the question, along with a doubt raised about the solution, and the new solution generated based on the doubt. You need to evaluate the effectiveness of the doubt, determining whether it effectively helped improve and refine the original solution.
Respond with only a score number from 0 to 10

<Question>:
{question}

<Solution>:
{solution}

<Doubt>:
{reflection}

<New Solution>:
{new_solution}

<Score>:"""
