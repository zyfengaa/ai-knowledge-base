"""
多智能体协作演示 — AutoGen GroupChat
"""
import os
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {
    "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}],
}

coder = AssistantAgent("coder", system_message="Python 工程师，写高质量代码。", llm_config=llm_config)
reviewer = AssistantAgent("reviewer", system_message="代码审查者，检查正确性和风格。", llm_config=llm_config)
tester = AssistantAgent("tester", system_message="测试工程师，编写测试用例。", llm_config=llm_config)

user = UserProxyAgent("user", human_input_mode="NEVER", code_execution_config={"use_docker": False})
groupchat = GroupChat(agents=[coder, reviewer, tester, user], messages=[], max_round=6)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

if __name__ == "__main__":
    user.initiate_chat(manager, message="写一个斐波那契数列函数，要求高效")
