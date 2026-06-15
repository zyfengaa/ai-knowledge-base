"""
MCP 风格工具抽象演示
"""
from typing import Any, Dict

class MCPTool:
    """工具基类，所有工具实现这个接口"""
    name: str = ""
    description: str = ""
    input_schema: Dict = {}

    def execute(self, args: Dict[str, Any]) -> str:
        raise NotImplementedError

class CalculatorTool(MCPTool):
    name = "calculator"
    description = "四则运算"
    input_schema = {"type": "object", "properties": {
        "a": {"type": "number"}, "b": {"type": "number"},
        "op": {"type": "string", "enum": ["+", "-", "*", "/"]}
    }, "required": ["a", "b", "op"]}

    def execute(self, args):
        return str(eval(f"{args['a']} {args['op']} {args['b']}"))

class WeatherTool(MCPTool):
    name = "get_weather"
    description = "查询天气"
    input_schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}

    def execute(self, args):
        temps = {"北京": 28, "上海": 32, "广州": 35}
        t = temps.get(args["city"], 25)
        return f"{args['city']} {t}degC"

class MCPServer:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def list_tools(self):
        return [{"name": t.name, "description": t.description, 
                 "input_schema": t.input_schema} for t in self._tools.values()]

    def execute(self, name: str, args: Dict) -> str:
        tool = self._tools.get(name)
        if not tool: return f"Unknown tool: {name}"
        return tool.execute(args)

if __name__ == "__main__":
    server = MCPServer()
    server.register(CalculatorTool())
    server.register(WeatherTool())
    print("Tools:", [t["name"] for t in server.list_tools()])
    print(server.execute("calculator", {"a": 25, "b": 48, "op": "*"}))
    print(server.execute("get_weather", {"city": "北京"}))
