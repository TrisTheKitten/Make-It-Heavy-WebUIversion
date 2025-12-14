import json
import yaml
import base64
from abc import ABC, abstractmethod
from typing import List, Optional
from openai import OpenAI
from tools import discover_tools


class BaseAgent(ABC):
    """Abstract base class for LLM agents"""
    
    def __init__(self, config_path="config.yaml", silent=False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.silent = silent
        self.discovered_tools = discover_tools(self.config, silent=self.silent)
        self.tool_mapping = {name: tool.execute for name, tool in self.discovered_tools.items()}
    
    @abstractmethod
    def call_llm(self, messages):
        """Make API call to the LLM provider"""
        pass
    
    @abstractmethod
    def handle_tool_call(self, tool_call):
        """Handle a tool call and return the result"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name"""
        pass
    
    @abstractmethod
    def run(self, user_input: str, images: Optional[List[dict]] = None) -> str:
        """Run the agent with user input and optional images"""
        pass
    
    @abstractmethod
    def remove_tool(self, tool_name: str):
        """Remove a tool by name from the agent"""
        pass


class OpenRouterAgent(BaseAgent):
    def __init__(self, config_path="config.yaml", silent=False):
        super().__init__(config_path, silent)
        
        self.client = OpenAI(
            base_url=self.config['openrouter']['base_url'],
            api_key=self.config['openrouter']['api_key']
        )
        self.tools = [tool.to_openrouter_schema() for tool in self.discovered_tools.values()]
    
    def get_provider_name(self) -> str:
        return "OpenRouter"
    
    def get_model_name(self) -> str:
        return self.config['openrouter']['model']
    
    def remove_tool(self, tool_name: str):
        self.tools = [t for t in self.tools if t.get('function', {}).get('name') != tool_name]
        self.tool_mapping = {n: f for n, f in self.tool_mapping.items() if n != tool_name}
    
    def call_llm(self, messages):
        """Make OpenRouter API call with tools"""
        try:
            response = self.client.chat.completions.create(
                model=self.config['openrouter']['model'],
                messages=messages,
                tools=self.tools
            )
            return response
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def handle_tool_call(self, tool_call):
        """Handle a tool call and return the result message"""
        try:
            # Extract tool name and arguments
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Call appropriate tool from tool_mapping
            if tool_name in self.tool_mapping:
                tool_result = self.tool_mapping[tool_name](**tool_args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            # Return tool result message
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_result)
            }
        
        except Exception as e:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps({"error": f"Tool execution failed: {str(e)}"})
            }
    
    def _build_user_content(self, user_input: str, images: Optional[List[dict]] = None):
        """Build user content with optional images for OpenAI-compatible API"""
        if not images:
            return user_input
        
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime_type']};base64,{img['data']}"
                }
            })
        content.append({"type": "text", "text": user_input})
        return content
    
    def run(self, user_input: str, images: Optional[List[dict]] = None):
        """Run the agent with user input and return FULL conversation content"""
        user_content = self._build_user_content(user_input, images)
        
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        full_response_content = []
        
        # Implement agentic loop from OpenRouter docs
        max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if not self.silent:
                print(f"🔄 Agent iteration {iteration}/{max_iterations}")
            
            # Call LLM
            response = self.call_llm(messages)
            
            # Add the response to messages
            assistant_message = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })
            
            # Capture assistant content for full response
            if assistant_message.content:
                full_response_content.append(assistant_message.content)
            
            # Check if there are tool calls
            if assistant_message.tool_calls:
                if not self.silent:
                    print(f"🔧 Agent making {len(assistant_message.tool_calls)} tool call(s)")
                # Handle each tool call
                task_completed = False
                for tool_call in assistant_message.tool_calls:
                    if not self.silent:
                        print(f"   📞 Calling tool: {tool_call.function.name}")
                    tool_result = self.handle_tool_call(tool_call)
                    messages.append(tool_result)
                    
                    # Check if this was the task completion tool
                    if tool_call.function.name == "mark_task_complete":
                        task_completed = True
                        if not self.silent:
                            print("✅ Task completion tool called - exiting loop")
                        # Return FULL conversation content, not just completion message
                        return "\n\n".join(full_response_content)
                
                # If task was completed, we already returned above
                if task_completed:
                    return "\n\n".join(full_response_content)
            else:
                if not self.silent:
                    print("💭 Agent responded without tool calls - continuing loop")
            
            # Continue the loop regardless of whether there were tool calls or not
        
        # If max iterations reached, return whatever content we gathered
        return "\n\n".join(full_response_content) if full_response_content else "Maximum iterations reached. The agent may be stuck in a loop."


class GeminiAgent(BaseAgent):
    def __init__(self, config_path="config.yaml", silent=False):
        super().__init__(config_path, silent)
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=self.config['gemini']['api_key'])
        self.tools = self._build_gemini_tools()
    
    def _build_gemini_tools(self):
        function_declarations = []
        for tool in self.discovered_tools.values():
            schema = tool.to_openrouter_schema()
            func_decl = {
                "name": schema["function"]["name"],
                "description": schema["function"]["description"],
                "parameters": schema["function"]["parameters"]
            }
            function_declarations.append(func_decl)
        return self.types.Tool(function_declarations=function_declarations)
    
    def get_provider_name(self) -> str:
        return "Gemini"
    
    def get_model_name(self) -> str:
        return self.config['gemini']['model']
    
    def remove_tool(self, tool_name: str):
        remaining_tools = {n: t for n, t in self.discovered_tools.items() if n != tool_name}
        self.discovered_tools = remaining_tools
        self.tool_mapping = {n: f for n, f in self.tool_mapping.items() if n != tool_name}
        self.tools = self._build_gemini_tools()
    
    def call_llm(self, contents, config):
        try:
            return self.client.models.generate_content(
                model=self.config['gemini']['model'],
                contents=contents,
                config=config
            )
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def handle_tool_call(self, function_call):
        try:
            tool_name = function_call.name
            tool_args = dict(function_call.args) if function_call.args else {}
            
            if tool_name in self.tool_mapping:
                tool_result = self.tool_mapping[tool_name](**tool_args)
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            
            return tool_result
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    def _build_content_parts(self, user_input: str, images: Optional[List[dict]] = None):
        """Build content parts with optional images for Gemini API"""
        parts = []
        
        if images:
            for img in images:
                image_bytes = base64.b64decode(img['data'])
                parts.append(
                    self.types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=img['mime_type']
                    )
                )
        
        parts.append(self.types.Part.from_text(text=user_input))
        return parts
    
    def run(self, user_input: str, images: Optional[List[dict]] = None):
        system_instruction = self.config['system_prompt']
        config = self.types.GenerateContentConfig(
            tools=[self.tools],
            system_instruction=system_instruction
        )
        
        parts = self._build_content_parts(user_input, images)
        contents = [self.types.Content(role="user", parts=parts)]
        full_response_content = []
        max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            if not self.silent:
                print(f"🔄 Agent iteration {iteration}/{max_iterations}")
            
            response = self.call_llm(contents, config)
            
            if not response.candidates:
                break
            
            response_content = response.candidates[0].content
            contents.append(response_content)
            
            if response.text:
                full_response_content.append(response.text)
            
            function_calls = []
            for part in response_content.parts or []:
                if part.function_call:
                    function_calls.append(part.function_call)
            
            if function_calls:
                if not self.silent:
                    print(f"🔧 Agent making {len(function_calls)} tool call(s)")
                
                function_response_parts = []
                task_completed = False
                
                for fc in function_calls:
                    if not self.silent:
                        print(f"   📞 Calling tool: {fc.name}")
                    
                    result = self.handle_tool_call(fc)
                    
                    if fc.name == "mark_task_complete":
                        task_completed = True
                        if not self.silent:
                            print("✅ Task completion tool called - exiting loop")
                    
                    function_response_parts.append(
                        self.types.Part.from_function_response(
                            name=fc.name,
                            response={"result": result}
                        )
                    )
                
                contents.append(self.types.Content(role="user", parts=function_response_parts))
                
                if task_completed:
                    return "\n\n".join(full_response_content)
            else:
                if not self.silent:
                    print("💭 Agent responded without tool calls - continuing loop")
        
        return "\n\n".join(full_response_content) if full_response_content else "Maximum iterations reached."


def create_agent(config_path="config.yaml", silent=False) -> BaseAgent:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    provider = config.get('provider', 'openrouter').lower()
    
    if provider == 'gemini':
        return GeminiAgent(config_path, silent)
    else:
        return OpenRouterAgent(config_path, silent)