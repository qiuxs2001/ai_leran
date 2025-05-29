# 导入所需的库
import json
import chromadb
import os
from openai import OpenAI 
from dotenv import find_dotenv, load_dotenv
from modelscope import AutoTokenizer, AutoModel
import torch
from zhipuai import ZhipuAI
import gradio as gr
import base64

def parse_json(file_path):
    """解析JSON文件,提取问答对
    Args:
        file_path (str): JSON文件路径,包含问答对数据
    Returns:
        tuple: (keywords, contents)
            - keywords (list): 问题列表
            - contents (list): 答案列表,每个答案为dict格式
    """
    keywords, contents = [], []  # 初始化问答对存储列表
    
    with open(file_path, "r", encoding="utf-8") as file:
        data_source = json.load(file)  # 读取JSON数据
        for item in data_source:  # 遍历每条问答对
            text = item['k_qa_content']  # 获取问答内容
            key, content = text.split("#\n")  # 分割问题和答案
            keywords.append(key)  # 存储问题
            contents.append({"content": content})  # 存储答案
    return keywords, contents

# def api_embedding(texts, model_name):
#     """使用OpenAI API生成文本向量表示
#     Args:
#         texts (list): 待处理的文本列表
#         model_name (str): 使用的模型名称
#     Returns:
#         list: 文本的向量表示列表
#     """
#     # 初始化API客户端
#     client = OpenAI(
#         api_key=os.getenv("api_key"),
#         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
#     )
#     embeddings = []
    
#     # 处理每条文本并生成向量
#     for input_text in texts:
#         completion = client.embeddings.create(
#             model=model_name,
#             input=input_text,
#             dimensions=768  # 设置向量维度
#         )
#         embedding = completion.data[0].embedding
#         embeddings.append(embedding)
#         return embedding

def local_embedding(sentences):
    """使用本地BGE模型生成文本向量
    Args:
        sentences (list): 待处理的文本列表
    Returns:
        list: 文本的向量表示列表
    """
    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()  # 设置为评估模式

    # 文本编码和向量生成
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():  # 禁用梯度计算
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]  # 获取[CLS]标记的输出
    
    return sentence_embeddings.numpy().tolist()  # 转换为Python列表

def llm_chat(message, history=[]):
    """调用智谱AI的GLM模型进行对话
    Args:
        message (str): 用户输入的消息文本
        history (list): 对话历史记录
    Returns:
        dict: 模型的回复消息
    """
    # 加载环境变量
    load_dotenv(".env")
    
    # 创建智谱AI客户端
    client = ZhipuAI(api_key=os.environ["zhipu_key"])
    
    # 构建消息列表，包含历史对话
    messages = []
    # 只保留最近5轮对话
    for h in history[-10:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})
    
    # 发送对话请求
    response = client.chat.completions.create(
        model="glm-4-flash-250414",  # 使用GLM-4模型
        messages=messages
    )
    return response.choices[0].message

def fetch_text_from_image(img_path):
    """从图片中提取文本内容
    Args:
        img_path (str): 图片文件路径
    Returns:
        str: 提取的文本内容
    """
    # 读取并编码图片
    with open(img_path, "rb") as image_file:
        image_base = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 初始化智谱AI客户端
    client = ZhipuAI(api_key=os.environ["zhipu_key"])
    
    # 创建多模态对话请求
    response = client.chat.completions.create(
        model="glm-4v-plus-0111",  # 使用支持图文理解的模型
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base  # 传入Base64编码的图片
                        }
                    },
                    {
                        "type": "text",
                        "text": "从图像中提取文本和代码"  # OCR提取指令
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

def chat_msg(message, history):
    """处理用户的图文对话消息
    Args:
        message (dict): 用户输入的消息,包含文本和图片
        history (list): 对话历史记录
    Returns:
        list: 更新后的对话历史
    """
    # 处理图片文件
    img_text = "无图片文本"
    if message["files"]:
        img_path = message["files"][0]
        img_text = fetch_text_from_image(img_path)
        history.append({"role": "user", "content": f"[用户上传了图片]\n提取的文本：{img_text}"})

    # 处理文本消息
    if message["text"] is not None:
        current_message = message["text"]
        
        # 生成问题的向量表示
        question = current_message + " " + img_text
        q_emb = local_embedding([question])
        
        # 在向量数据库中查询相似内容
        collection = client.get_collection('my_collection')
        result = collection.query(query_embeddings=q_emb, n_results=1)
        
        # 提取查询结果
        content = ""
        if len(result['metadatas']) > 0:
            content = result['metadatas'][0][0]['content']
        
        # 构建提示词，包含历史对话
        recent_history = history[-6:]  # 保留最近3轮对话
        history_text = "\n".join([
            f"{'用户' if h['role']=='user' else 'AI助手'}: {h['content']}"
            for h in recent_history
        ])
        
        prompt = f"""你是一个精通python语言编程的专家，请基于以下信息回答问题：

当前问题: {current_message}
图片内容: {img_text if img_text != "无图片文本" else "无"}
对话历史:
{history_text}

参考资料: {content}

请用markdown格式回复，确保代码部分使用```python进行标注。"""
        
        # 获取AI回复并更新历史
        answer = llm_chat(prompt, history)
        history.append({"role": "user", "content": current_message})
        history.append({"role": "assistant", "content": answer.content})
    
    return history

if __name__ == '__main__':
    # 初始化环境和数据
    load_dotenv(find_dotenv())
    keywords, contents = parse_json("data_source.json")
    embeddings = local_embedding(keywords)
    
    # 设置ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_or_create_collection("my_collection")
    
    # 首次运行时初始化向量数据库
    if not os.path.exists('chroma'):
        # 生成文档ID
        ids = [f"id{i}" for i in range(len(keywords))]
        
        # 添加数据到向量数据库
    # 创建Gradio界面
    with gr.Blocks() as demo:
        # 初始化聊天组件
        chatbot = gr.Chatbot(
            type="messages",
            height=600,
            show_copy_button=True
        )
        # 创建多模态输入框
        tbox = gr.MultimodalTextbox(
            sources=['upload'],
            file_count="single", 
            file_types=['image'],
            container=False,
            scale=7
        )
        # 添加清除按钮
        clear = gr.Button("清除对话")
        
        # 绑定事件
        tbox.submit(fn=chat_msg, inputs=[tbox, chatbot], outputs=chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
        
        # 启动界面
        demo.launch(share=False, debug=True)