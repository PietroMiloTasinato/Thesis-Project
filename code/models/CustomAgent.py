import ollama
import wikipedia
#import .CustomClient

MODEL = "internlm/interns1:mini" #"llama3.1" "qwen3.5:27b" "gemma4:26b" "internlm/interns1:mini"
HOST = "http://localhost:11434"
MAX_TOOL_ROUNDS = 5  # safety limit so a looping model can't run forever

client = ollama.Client(host=HOST)

'''
def search(query: str) -> str:
    #Find more information on a specific topic on Wikipedia.

    Args:
        query: The topic to look up.

    Returns:
        The beginning of the best matching Wikipedia article.
    #
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return f"No Wikipedia article found for '{query}'."
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.content[:4000]
    except wikipedia.exceptions.DisambiguationError as e:
        return f"'{query}' is ambiguous. Options include: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'."
    except Exception as e:  # network errors, etc.
        return f"Wikipedia lookup failed: {e}"

'''
# ollama client class that implements customClient abs
class OllamaClient:

    def __init__(self, model=MODEL, host=HOST, temperature=0.3, num_ctx=8192,
                 tools=None, system_prompt=None):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx  # Ollama's default context window is small
        self.system_prompt = system_prompt  # default, can be overridden per call
        self.tools = {fn.__name__: fn for fn in (tools or [])}
        self._client = ollama.Client(host=host)

    def call_single_messages(self, user_prompt: str = None, system_prompt: list = [],
             temperature: float = None, max_tokens: int = None, **_ignored) -> str:
        #Send one independent query to the LLM and return the reply text
        system = system_prompt if system_prompt is not None else self.system_prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        #return self._run(messages, temperature, max_tokens)
        return self.call(messages=messages, temperature=temperature, max_tokens=max_tokens)

    def create_agent(self, system_prompt: str) -> "CustomAgent":
        return CustomAgent(self, system_prompt)

    # Chat loop: run any tool calls the model requests, return the final reply.
    def call(self, messages: list, temperature=None, max_tokens=None) -> str:

        options = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._client.chat(
                model=self.model,
                messages=messages,
                tools=list(self.tools.values()) or None,
                options=options,
                keep_alive= -1
            )
            message = response.message
            #print("AGENT RESPONSE | " + str(message))
            messages.append(message)

            if not message.tool_calls:
                return message.content

            for call in message.tool_calls:
                name = call.function.name
                tool = self.tools.get(name)
                if tool is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = str(tool(**call.function.arguments))
                    except Exception as e:
                        result = f"Tool error: {e}"
                messages.append({"role": "tool", "tool_name": name, "content": result})

        return "Stopped: too many consecutive tool calls"


class CustomAgent:

    def __init__(self, client, system_prompt: str):
        self.client = client
        self.messages = [{"role": "system", "content": system_prompt}]

    def call(self, prompt: str, temperature: float = None,
             max_tokens: int = None, **_ignored) -> str:
        self.messages.append({"role": "user", "content": prompt})
        return self.client._run(self.messages, temperature, max_tokens)

    def reset(self):
        self.messages = self.messages[:1]

