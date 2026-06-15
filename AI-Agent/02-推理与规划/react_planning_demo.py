"""
ReAct + 自省 Agent 演示 — OpenAI 版
运行: export OPENAI_API_KEY="sk-xxx"
"""
import json, os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tools = [{"type": "function", "function": {
    "name": "search", "description": "搜索信息",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
}}]

def agent_react(task):
    memory = [{"role": "user", "content": task}]
    for step in range(4):
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=memory, tools=tools, tool_choice="auto")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(f"[Answer]: {msg.content}")
            return
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"[Action]: {tc.function.name}({args})")
            memory.append(msg)
            memory.append({"role": "tool", "tool_call_id": tc.id, "content": "OK"})
            memory.append({"role": "user", "content": "反思当前进展，然后继续。"})
    print("Max steps reached")

if __name__ == "__main__":
    agent_react("比较 Python 和 JavaScript 的优劣")
