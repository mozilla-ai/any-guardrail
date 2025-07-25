# Centralized constants for the any_guardrail package

# Package name for guardrail implementations
GUARDRAIL_PKG = 'any_guardrail.guardrails'

# Common label strings
LABEL_UNSAFE = "UNSAFE"
LABEL_SAFE = "SAFE"

# Common model output labels (case-sensitive)
LABEL_INJECTION_UPPER = "INJECTION"
LABEL_INJECTION_LOWER = "injection"
LABEL_JAILBREAK = "jailbreak"
LABEL_UNSAFE_LOWER = "unsafe"
LABEL_NO = "No"
LABEL_YES = "Yes"

# Default threshold values (can be overridden by user in class instances)
DEFAULT_THRESHOLD = 0.5

# DuoGuard categories
DUOGUARD_CATEGORIES = [
    "Violent crimes",
    "Non-violent crimes",
    "Sex-related crimes",
    "Child sexual exploitation",
    "Specialized advice",
    "Privacy",
    "Intellectual property",
    "Indiscriminate weapons",
    "Hate",
    "Suicide and self-harm",
    "Sexual content",
    "Jailbreak prompts",
]

# System prompt for ShieldGemma
SYSTEM_PROMPT_SHIELD_GEMMA = """You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Question: {user_prompt}
<end_of_turn>

Our safety principle is defined in the below:

{safety_policy}

Does the human question violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly.
"""

# System prompt for GLIDER
SYSTEM_PROMPT_GLIDER = """
Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:

1. Understand the text tags, pass criteria and rubric thoroughly.
2. Review the finer details of the text and the rubric.
3. Compare the tags to be evaluated to the score descriptions in the rubric.
4. Pay close attention to small details that might impact the final score and form accurate associations between tags and pass criteria.
5. Write a detailed reasoning justifying your evaluation in a bullet point format. 
6. The reasoning must summarize the overall strengths and weaknesses of the output while quoting exact phrases from the output wherever required.
7. Output a list of words or phrases that you believe are the most important in determining the score.
8. Assign a final score based on the scoring rubric.

Data to evaluate:
{data}

Pass Criteria:
{pass_criteria}

Rubric:
{rubric}

Your output must be in the following format:
<reasoning>
[Detailed reasoning justifying your evaluation in a bullet point format according to the specifics defined above]
</reasoning>
<highlight>
[List of words or phrases that you believe are the most important in determining the score]
</highlight>
<score>
[The final integer score assigned based on the scoring rubric]
</score>
""" 