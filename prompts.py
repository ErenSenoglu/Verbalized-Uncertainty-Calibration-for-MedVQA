VANILLA_TEMPLATE = (
    "<bos><start_of_turn>user\n"
    "<start_of_image>{question}<end_of_turn>\n"
    "Choose exactly one answer from the four options below.\n"
    "Options:\n"
    "- A: {option_a}\n"
    "- B: {option_b}\n"
    "- C: {option_c}\n"
    "- D: {option_d}\n\n"
    "Based on the image, provide the most likely medical finding or answer concisely by following the output format below.\n"
    "{STEER_INSTRUCTION}\n"
    "Output format:\n"
    "Rationale: <1–2 concise sentences>\n"
    "Answer: <LETTER> — <EXACT OPTION TEXT>\n"
    "After the answer please provide your confidence on your answer in the following format on a scale of 1 to 10:\n"
    "Confidence: <1-10>\n"
    "<start_of_turn>model\n"
)
VANILLA_TEMPLATE_5 = (
    "<bos><start_of_turn>user\n"
    "<start_of_image>{question}<end_of_turn>\n"
    "Choose exactly one answer from the five options below.\n"
    "Options:\n"
    "- A: {option_a}\n"
    "- B: {option_b}\n"
    "- C: {option_c}\n"
    "- D: {option_d}\n"
    "- E: {option_e}\n\n"
    "Based on the image, provide the most likely medical finding or answer concisely by following the output format below.\n"
    "{STEER_INSTRUCTION}\n"
    "Output format:\n"
    "Rationale: <1–2 concise sentences>\n"
    "Answer: <LETTER> — <EXACT OPTION TEXT>\n"
    "After the answer please provide your confidence on your answer in the following format on a scale of 1 to 10:\n"
    "Confidence: <1-10>\n"
    "<start_of_turn>model\n"
)

TOPK_TEMPLATE = """
<bos><start_of_turn>user
<start_of_image>{question}<end_of_turn>
- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}

Provide your 2 best guesses and your confidence that each is correct.
After each answer please provide your confidence on a scale of 0 to 100.
Give ONLY the guesses and confidence scores, no explanation is required.
Option text and option letter MUST match with the provided options.
The guesses MUST be different.
Follow this format for your answers:
Guess: <option_text> — <option_letter>
Confidence: <0-100>
Guess: <option_text> — <option_letter>
Confidence: <0-100>
<start_of_turn>model
"""

TOPK_TEMPLATE_5 = """
<bos><start_of_turn>user
<start_of_image>{question}<end_of_turn>
- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}
- E: {option_e}

Provide your 2 best guesses and your confidence that each is correct.
After each answer please provide your confidence on a scale of 0 to 100.
Give ONLY the guesses and confidence scores, no explanation is required.
Option text and option letter MUST match with the provided options.
The guesses MUST be different.
Follow this format for your answers:
Guess: <option_text> — <option_letter>
Confidence: <0-100>
Guess: <option_text> — <option_letter>
Confidence: <0-100>
<start_of_turn>model
"""

QWEN_VANILLA_TEMPLATE = """{question}

Choose exactly one answer from the four options below.
Options:
- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}

Based on the image, provide the most likely medical finding or answer concisely by following the output format below.
{STEER_INSTRUCTION}
Output format:
Rationale: <1–2 concise sentences>
Answer: <LETTER> — <EXACT OPTION TEXT>
After the answer please provide your confidence on your answer in the following format on a scale of 0 to 9:
Confidence: <0-9>
"""

QWEN_VANILLA_TEMPLATE_5 = """{question}

Choose exactly one answer from the five options below.
Options:
- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}
- E: {option_e}

Based on the image, provide the most likely medical finding or answer concisely by following the output format below.
{STEER_INSTRUCTION}
Output format:
Rationale: <1–2 concise sentences>
Answer: <LETTER> — <EXACT OPTION TEXT>
After the answer please provide your confidence on your answer in the following format on a scale of 0 to 9:
Confidence: <0-9>
"""

QWEN_TOPK_TEMPLATE = """{question}

- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}

Provide your 2 best guesses and your confidence that each is correct.
After each answer please provide your confidence on a scale of 0 to 100.
Give ONLY the guesses and confidence scores, no explanation is required.
Option text and option letter MUST match with the provided options.
The guesses MUST be different.
Follow this format for your answers:
Guess: <option_text> — <option_letter>
Confidence: <0-100>
Guess: <option_text> — <option_letter>
Confidence: <0-100>
"""

QWEN_TOPK_TEMPLATE_5 = """{question}

- A: {option_a}
- B: {option_b}
- C: {option_c}
- D: {option_d}
- E: {option_e}

Provide your 2 best guesses and your confidence that each is correct.
After each answer please provide your confidence on a scale of 0 to 100.
Give ONLY the guesses and confidence scores, no explanation is required.
Option text and option letter MUST match with the provided options.
The guesses MUST be different.
Follow this format for your answers:
Guess: <option_text> — <option_letter>
Confidence: <0-100>
Guess: <option_text> — <option_letter>
Confidence: <0-100>
"""
