from openai import OpenAI
from src.config import settings

class Generator:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.llm_model
    
    def generate_answer(self, query, context_docs):
        """根据检索到的上下文生成答案"""
        if not context_docs:
            return "抱歉，我没有找到相关的信息来回答您的问题。"
        
        # 构建上下文
        context = ""
        for i, doc in enumerate(context_docs):
            source = doc["metadata"]["source"]
            page = doc["metadata"]["page"]
            context += f"【来源{i+1}：{source}，第{page}页】\n{doc['content']}\n\n"
        
        # 构建Prompt
        prompt = f"""
你是一个专业的知识库助手，请严格按照以下规则回答问题：
1. 只能使用上面提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确回答"无法解答"
3. 答案必须准确、简洁、客观
4. 每个答案都必须标注来源，格式为：[来源1]、[来源2]...

问题：{query}

上下文：
{context}

答案：
"""
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens
        )
        
        return response.choices[0].message.content