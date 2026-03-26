import re
from typing import Any

class Segmenter:
    def __init__(self):
        self.math_block_pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])')
        
    def segment(self, text: str) -> list[dict[str, Any]]:
        """
        返回包含坐标信息的字典列表：[{"text": "...", "start": 0, "end": 45}, ...]
        """
        # 1. 记录被保护的公式区块坐标 (避免内部的 \n\n 被切断)
        math_spans =[]
        for match in self.math_block_pattern.finditer(text):
            math_spans.append((match.start(), match.end()))
            
        def is_in_math(pos):
            for ms, me in math_spans:
                if ms <= pos < me: return True
            return False

        # 2. 寻找合法的 \n\n 切分点
        split_points = [0]
        for match in re.finditer(r'\n{2,}', text):
            if not is_in_math(match.start()):
                split_points.extend([match.start(), match.end()])
        split_points.append(len(text))
        
        # 3. 提取初始区块并记录精确坐标
        chunks_info =[]
        for i in range(0, len(split_points)-1, 2):
            s, e = split_points[i], split_points[i+1]
            chunk_str = text[s:e]
            if not chunk_str.strip():
                continue
            
            # 缩紧坐标，去掉前后空白字符
            real_s = s + (len(chunk_str) - len(chunk_str.lstrip()))
            real_e = e - (len(chunk_str) - len(chunk_str.rstrip()))
            clean_str = text[real_s:real_e]
            
            # 过滤纯分割线 (---)
            if re.fullmatch(r'-+', clean_str):
                continue
                
            chunks_info.append({"text": clean_str, "start": real_s, "end": real_e})
            
        # 4. Smart Merging (智能合并的同时合并坐标)
        merged_info =[]
        for info in chunks_info:
            if not merged_info:
                merged_info.append(info)
                continue
                
            prev_info = merged_info[-1]
            curr_text, prev_text = info["text"], prev_info["text"]
            
            # 合并规则 1：公式吸附
            # 如果当前块是一个纯公式块，它大概率属于上一句的推导结果，必须合并
            is_math_block: bool = curr_text.startswith('$$') or curr_text.startswith('\\[')
            
            # 合并规则 2：标题/步骤吸附
            # 如果上一个块是 "### Step X" 或 "**Step X:**" 这种独立标题，合并当前内容
            is_prev_heading: bool = bool(re.match(r'^(#{1,6}\s+.*|\*\*Step\s*\d+.*\*\*)$', prev_text, re.IGNORECASE))
            
            # 合并规则 3：未完结语气吸附
            # 如果上一句以冒号(:)、逗号(,)结尾
            expects_continuation: bool = prev_text.endswith(':') or prev_text.endswith(',') \
                                      or prev_text.endswith('：') or prev_text.endswith('，')
            
            # 合并规则 4：列表项吸附
            # 如果当前是列表(- 或 *)，且上一句是简短的引导语 (比如 "We are given:")
            is_current_list: bool = curr_text.startswith('- ') or curr_text.startswith('* ')
            is_prev_short_intro: bool = len(prev_text.split('\n')) == 1 and len(prev_text) < 80
            
            if is_math_block or is_prev_heading or expects_continuation or (is_current_list and is_prev_short_intro):
                # 合并坐标和文本 (终点拉伸到当前块的终点)
                merged_info[-1]["text"] = text[prev_info["start"]:info["end"]].strip()
                merged_info[-1]["end"] = info["end"]
            else:
                merged_info.append(info)
                
        return merged_info

def segment(texts: list[str]) -> list[list[dict[str, Any]]]:
    segmenter = Segmenter()
    return [segmenter.segment(text) for text in texts]

def get_token_char_spans(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """
    输入: token ID 列表
    输出: [(start_char_idx, end_char_idx), ...] 长度等于 token 数量
    """
    # 构造逐级递增的前缀序列 [token0], [token0, token1], ...
    prefix_ids = [token_ids[:i] for i in range(1, len(token_ids) + 1)]
    
    # 批量解码
    decoded_prefixes = tokenizer.batch_decode(prefix_ids, skip_special_tokens=True)
    
    spans =[]
    prev_len = 0
    for text in decoded_prefixes:
        curr_len = len(text)
        spans.append((prev_len, curr_len))
        prev_len = curr_len
        
    return spans

def find_token_idx_for_char_idx(token_ids: list[int], char_idx: int, tokenizer) -> int:
    """
    通过二分查找，寻找第一个使得 解码后字符串长度 >= char_idx 的 Token 索引。
    复杂度: O(log N) 次 decode
    """
    low, high = 0, len(token_ids)
    ans = high
    while low <= high:
        mid = (low + high) // 2
        text = tokenizer.decode(token_ids[:mid], skip_special_tokens=True)
        if len(text) >= char_idx:
            ans = mid
            high = mid - 1  # 满足条件，继续向左尝试，以防有不占长度的特殊Token
        else:
            low = mid + 1
    return ans