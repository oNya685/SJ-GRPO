from verl import DataProto

def judge_template(prompts, chunks_info_list, scores):
    outputs = [chunks_info["text"] for chunks_info in chunks_info_list]

    judge_prompts = []
    for prompt, output, score in zip(prompts, outputs, scores):
        judge_prompts.append(f"Question: {prompt}\nAnswer: {output}\nResult: {score}\n---\n")
    return DataProto().from_dict({"prompts": judge_prompts})