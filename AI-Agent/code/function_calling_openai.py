"""
OpenAI Function Calling Demo — 最简单的 Agent 实现
功能：LLM 自动决定是否调用 API，调用天气/计算器

运行: export OPENAI_API_KEY="sk-xxx"
      python code/function_calling_openai.py
"""
import json, os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === 定义工具 ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "四则运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"}, "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]}
                },
                "required": ["a", "b", "op"]
            }
        }
    }
]

# === 工具实现 ===
def get_weather(city: str) -> str:
    """模拟天气查询"""
    import random
    temps = {"北京": 28, "上海": 32, "广州": 35, "深圳": 33}
    t = temps.get(city, random.randint(15, 35))
    return f"{city} 当前温度 {t}°C，天气晴"

def calculator(a: float, b: float, op: str) -> float:
    return eval(f"{a} {op} {b}")

def call_function(name: str, args: dict) -> str:
    if name == "get_weather": return get_weather(**args)
    if name == "calculator": return calculator(**args)
    return f"未知工具: {name}"

# === Agent 主循环（ReAct 风格）===
def agent_loop(user_input: str, max_steps: int = 5):
    messages = [{"role": "user", "content": user_input}]
    
    for step in range(max_steps):
        print(f"\n{'='*40}\n[Step {step+1}]")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        
        if not msg.tool_calls:
            # LLM 认为不需要调用工具，输出最终回答
            print(f"[Final]: {msg.content}")
            return msg.content
        
        # 处理工具调用
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = call_function(name, args)
            print(f"[Action]: {name}({args}) → {result}")
            
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })
    
    return "达到最大步数"

# === 测试 ===
if __name__ == "__main__":
    agent_loop("北京天气怎么样？然后帮我算 25 * 48")
