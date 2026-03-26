CHUNK_BEGIN_TEMPLATE = "<chunk {x} begin>"
CHUNK_END_TEMPLATE = "<chunk {x} end>"
SYSTEM_PROMPT = """
You are a judger and you should judge which chunks in the answer are key to make the QA pair correct or incorrect. You should give a score for each chunk, and the score should be 1 or 0, where 1 means the chunk is crucial, and 0 means the chunk is not important. The answer should be in JSON format:" \
{
    <chunk 0>: score,
    <chunk 1>: score,
    ...
}
where x is the index of the chunk in the answer, starting from 0. The chunks in the answer are marked with <chunk x begin> and <chunk x end>. You should only judge the chunks in the answer, and you should not judge any other part of what the user provides."
"""

def judge_template(tokenizer, prompts, chunks_info_list, scores) -> list[str]:
    # 处理 answers：为每个 chunk 添加 <chunk x begin> 和 <chunk x end>
    answers = []
    for chunks_info in chunks_info_list:
        # 使用 enumerate 获取索引 x
        formatted_chunks = [
            f"{CHUNK_BEGIN_TEMPLATE.format(x=x)}{chunk['text']}{CHUNK_END_TEMPLATE.format(x=x)}"
            for x, chunk in enumerate(chunks_info)
        ]
        # 将该样本的所有包裹后的 chunk 拼接成完整字符串
        answers.append("\n\n".join(formatted_chunks))

    # 确保 scores 是平铺的列表
    if hasattr(scores, "tolist"):
        scores = scores.flatten().tolist()

    judge_prompts = []
    for prompt, answer, score in zip(prompts, answers, scores):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse which chunks are crucial making the Answer {'correct' if score == 1 else 'incorrect'} in the following QA pair:\nQuestion: {prompt}\nAnswer: {answer}\n---\n"}
        ]
        judge_prompts.append(tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ))
    return judge_prompts