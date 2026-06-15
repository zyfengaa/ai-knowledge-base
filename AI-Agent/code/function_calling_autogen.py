"""
AutoGen Function Calling Demo — 框架版的 Agent 实现
AutoGen 自动管理多轮对话和工具调用循环

运行: export OPENAI_API_KEY="sk-xxx"
      pip install pyautogen
      python code/function_calling_autogen.py
"""
import os
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# === 配置 LLM ===
llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}],
    "timeout": 60,
}

# === 定义工具函数（AutoGen 通过 docstring 自动生成 Schema）===
def get_weather(city: str) -> str:
    """查询指定城市的天气预报
    
    Args:
        city (str): 城市名，如 "北京"
    
    Returns:
        str: 天气信息
    """
    import random
    temps = {"北京": 28, "上海": 32, "广州": 35}
    t = temps.get(city, random.randint(15, 35))
    return f"{city} 当前温度 {t}°C，天气晴"


def calculator(a: float, b: float, op: str) -> float:
    """四则运算计算器
    
    Args:
        a (float): 第一个数字
        b (float): 第二个数字
        op (str): 运算符，支持 + - * /
    
    Returns:
        float: 运算结果
    """
    return eval(f"{a} {op} {b}")


# === 创建 Agent ===
assistant = AssistantAgent(
    name="assistant",
    system_message="你是一个智能助手，可以使用工具查询天气和做计算。回答简洁准确。",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # 自动执行工具函数
    code_execution_config={"use_docker": False},
    function_map={
        "get_weather": get_weather,
        "calculator": calculator,
    },
)

# === 启动对话 ===
if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message="北京天气怎么样？然后帮我算 25 * 48",
        max_turns=5,
    )
