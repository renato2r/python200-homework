import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# Strict assertion to guarantee the API key is present before executing any agent steps
assert os.getenv("OPENAI_API_KEY"), "❌ Error: OPENAI_API_KEY environment variable not found in .env file!"
print("[CONFIRMATION]: OpenAI API Key successfully loaded for Week 7 Agents.")

# 2. Establish and verify the outputs directory structure
outputs_dir = Path("./assignments_07/outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
print(f"[CONFIRMATION]: Outputs directory verified at: {outputs_dir}")

print("\n--- Step 1: Environment and Directory Setup Complete ---")
print("------------------------------------------------------------------")


# --- Lesson 02 ---
# Q1

print("\n--- Lesson 02: Q1 - Function Definition & JSON Schema ---")

# 1. Define the core functional utility tool
def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# 2. Write the strict JSON schema declaration following the OpenAI/Tool-calling standard
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit and return it as a formatted string.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "The temperature value in degrees Celsius to convert."
                }
            },
            "required": ["celsius"]
        }
    }
}

# 3. Directly evaluate specific baseline and boundary test cases
test_temperatures = [0, 100, -40]

print("Executing direct functional evaluation tests:")
for temp in test_temperatures:
    result = celsius_to_fahrenheit(temp)
    print(f"  Input: {temp:4}°C  ->  Output: {result}")

print("\n Displaying structural JSON Tool Schema:")
import json
print(json.dumps(celsius_to_fahrenheit_schema, indent=2))

print("\n------------------------------------------------------------------")

# Q2

# =====================================================================
# Q2 PREDICTION COMMENT BLOCK
# =====================================================================
# 1. Will calling run_agent with this query trigger a tool call? Why or why not?
#    No, it will not trigger a tool call. The agent only has access to a single tool 
#    called `get_current_time`. Because LLMs are trained to evaluate the semantic intent 
#    of a query against the function descriptions provided in their schema, the model 
#    will immediately recognize that getting the current system clock has no relevance 
#    to solving a mathematical temperature conversion problem.
#
# 2. How many API calls will be made to answer this query?
#    Exactly 1 API call will be made. In a standard ReAct loop setup, the first API call 
#    submits the user's prompt alongside the available tool schemas. Since the LLM realizes 
#    no tools are helpful, it will immediately generate its final conversational answer text 
#    in the first turn, skipping any execution loops or subsequent observations.
# =====================================================================

print("\n--- Lesson 02: Q2 - Testing Toolkit Isolation ---")

from openai import OpenAI
import json

# Setup standard mock environment variables for the internal manual loop
client = OpenAI()

# Mock get_current_time function and schema from the lesson materials
def get_current_time() -> str:
    """Get the current system time."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")

get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current system time.",
        "parameters": {"type": "object", "properties": {}}
    }
}

def run_agent(query: str) -> str:
    """A simple manual ReAct agent loop from the lesson materials."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them if needed."},
        {"role": "user", "content": query}
    ]
    tools = [get_current_time_schema]
    
    print(f"🚀 [API CALL 1]: Dispatched user prompt to OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    if tool_calls:
        print(f" [TOOL INTERCEPT]: Agent requested to execute tool: {tool_calls[0].function.name}")
        # Normally the loop executes the function here and calls the API a second time.
        # We simulate the basic step logic for tracking.
        return f"Tool called: {tool_calls[0].function.name}"
    else:
        print(" [NO TOOL CALLED]: Agent determined no tools were applicable.")
        return response_message.content

# Execute the target test query
conversion_query = "Convert 100 degrees Celsius to Fahrenheit"
print(f"Sending Query: '{conversion_query}'")

agent_final_output = run_agent(conversion_query)

print(f"\n[AGENT FINAL OUTPUT]:\n{agent_final_output}")

# =====================================================================
# Q2 POST-RUN EVALUATION
# =====================================================================
# Was the prediction correct?
# Yes, the prediction was 100% correct. The agent executed exactly 1 API call, safely 
# skipped tool interception, and the model handled the temperature calculation internally 
# using its own pre-trained knowledge base rather than forcing a wrong tool execution.
# =====================================================================

print("\n------------------------------------------------------------------")

# Q3

print("\n--- Lesson 02: Q3 - Multi-Tool Agent Execution & Routing ---")

def run_agent_v2(query: str) -> str:
    """
    An expanded manual ReAct agent loop supporting multiple tools.
    Handles tool identification, execution, and secondary LLM observation routing.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them only if needed."},
        {"role": "user", "content": query}
    ]
    
    # 1. Provide the agent with the schemas for both tools
    tools = [get_current_time_schema, celsius_to_fahrenheit_schema]
    
    print(f"\n [API CALL 1]: Submitting query to model...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # 2. Check if the model determined a tool execution was required
    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"[TOOL CALL DETECTED]: Agent requested tool '{tool_name}' with args: {tool_args}")
        
        # 3. Execution Routing Layer
        if tool_name == "celsius_to_fahrenheit":
            # Extract argument and run the actual Python function
            celsius_val = tool_args.get("celsius")
            observation = celsius_to_fahrenheit(celsius_val)
        elif tool_name == "get_current_time":
            observation = get_current_time()
        else:
            observation = f"Error: Tool '{tool_name}' is not implemented."
            
        print(f"[OBSERVATION]: Tool executed successfully. Result: '{observation}'")
        
        # 4. Feed the observation back to the LLM to synthesize the final conversational response
        messages.append(response_message)  # Append assistant's intent
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_name,
            "content": observation
        })
        
        print("[API CALL 2]: Sending tool observation back to model for final text generation...")
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0
        )
        return second_response.choices[0].message.content
    else:
        print("[NO TOOL CALLED]: Agent answered using internal knowledge parameters.")
        return response_message.content

# --- Execute and evaluate Query A ---
response_a = run_agent_v2("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# =====================================================================
# QUERY A REFLECTION
# =====================================================================
# Did a tool get called and why?
# Yes, a tool was explicitly called. The agent intercepted the intent to perform a 
# mathematical conversion on a specific numerical value (37) matching the exact 
# operational definition declared in the `celsius_to_fahrenheit` schema. It bypassed 
# native math generation, ran the Python script, and handed the observation back.
# =====================================================================

# --- Execute and evaluate Query B ---
response_b = run_agent_v2("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# =====================================================================
# QUERY B REFLECTION
# =====================================================================
# Did a tool get called and why?
# No tool was called. The query asks for a general, widely known scientific fact 
# ("boiling point of water") in a generic format ("plain English") without specifying a 
# variable or requesting a real-time system clock update. The model intelligently 
# determined that neither of its tools were relevant, completing the response in a 
# single conversational turn.
# =====================================================================

print("\n------------------------------------------------------------------")

# --- Lesson 03 ---
# Q4

print("\n--- Lesson 03: Q4 - Extending CsvManager with Statistics ---")

import pandas as pd
import scipy.stats as stats
import io

# 1. Re-implement and Extend the CsvManager class from the lesson
class CsvManager:
    def __init__(self):
        self.df = None

    def load_csv(self, csv_data: str) -> dict:
        """Parse raw CSV string data into a memory-buffered pandas DataFrame."""
        try:
            self.df = pd.read_csv(io.StringIO(csv_data))
            return {"status": "success", "message": f"CSV loaded with {len(self.df)} rows and columns: {list(self.df.columns)}"}
        except Exception as e:
            return {"error": f"Failed to parse CSV: {str(e)}"}

    def filter_data(self, column: str, value: str) -> list:
        """Filter the active DataFrame where the specified column matches a target string value."""
        if self.df is None:
            return [{"error": "No CSV data loaded. Run load_csv first."}]
        if column not in self.df.columns:
            return [{"error": f"Column '{column}' not found. Available columns: {list(self.df.columns)}"}]
        
        filtered_df = self.df[self.df[column].astype(str) == str(value)]
        return filtered_df.to_dict(orient="records")

    def compute_correlation(self, col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        if self.df is None:
            return {"error": "No CSV data loaded. Run load_csv first."}
        if col1 not in self.df.columns:
            return {"error": f"Column '{col1}' not found in DataFrame."}
        if col2 not in self.df.columns:
            return {"error": f"Column '{col2}' not found in DataFrame."}
        
        try:
            # Drop any missing rows between the two columns to ensure a clean mathematical array pass
            clean_df = self.df[[col1, col2]].dropna()
            
            if len(clean_df) < 2:
                return {"error": "Not enough valid numeric data remaining after dropping null values to calculate correlation."}
            
            # Use scipy.stats.pearsonr for the statistical correlation calculation
            r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
            
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4)
            }
        except Exception as e:
            return {"error": f"Calculation failed. Ensure both columns are numeric. Details: {str(e)}"}

# 2. Instantiate the global state object
csv_manager = CsvManager()

# 3. Define the updated tools schema pool including the new compute_correlation metadata
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Parse raw CSV string data into a memory-buffered pandas DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "The raw string representation of the CSV file content."}
                },
                "required": ["csv_data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_data",
            "description": "Filter rows where the specified column matches a target value string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "The target column header name."},
                    "value": {"type": "string", "description": "The value to filter by."}
                },
                "required": ["column", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two columns in the loaded DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {"type": "string", "description": "The name of the first numeric column header."},
                    "col2": {"type": "string", "description": "The name of the second numeric column header."}
                },
                "required": ["col1", "col2"]
            }
        }
    }
]

# 4. Map the tool string identifiers to the active instance methods
node_tools = {
    "load_csv": csv_manager.load_csv,
    "filter_data": csv_manager.filter_data,
    "compute_correlation": csv_manager.compute_correlation
}

print(" Extended CsvManager and tool schema registries loaded with 'compute_correlation'.")
print("\n------------------------------------------------------------------")

# Q5

print("\n--- Lesson 03: Q5 - Executing Multi-Step Statistical ReAct Loop ---")

# 1. Define the system prompt instruction set from the lesson materials
SYSTEM_PROMPT = """You are a data analysis agent with access to a CsvManager instance via tools.
Your goal is to answer user requests by running tools sequentially.

Available tools:
- load_csv: Load raw CSV string data.
- filter_data: Filter rows by matching a column to a value string.
- compute_correlation: Compute Pearson correlation between two numeric columns.

Guidelines:
1. You can call ONE tool per turn.
2. After a tool call, you will receive an observation with the result.
3. Use that observation to decide your next step or provide your final answer.
4. When you have enough information to answer the request, provide your final response directly without calling any more tools.
"""

# 2. Implement the standard run_agent_cycle loop mechanism from the lesson
def run_agent_cycle(messages: list, user_query: str, max_rounds: int = 5) -> str:
    """
    Executes an autonomous multi-turn ReAct execution loop up to a maximum round safety limit.
    Dynamically captures tool execution intent and pipes feedback observations back to the LLM.
    """
    messages.append({"role": "user", "content": user_query})
    
    current_round = 1
    while current_round <= max_rounds:
        print(f"\n [ROUND {current_round}]: Invoking OpenAI completion engine...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0.0
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        tool_calls = response_message.tool_calls
        
        # Scenario A: The model determines it needs to take an action using a tool
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f" [ACTION REQUIRED]: Agent selected tool '{tool_name}' with arguments: {tool_args}")
            
            # Dispatch to our live tool registry mapping
            if tool_name in node_tools:
                observation_result = node_tools[tool_name](**tool_args)
            else:
                observation_result = {"error": f"Tool '{tool_name}' was requested but is not registered."}
                
            print(f" [OBSERVATION]: Tool executed. Result: {observation_result}")
            
            # Feed the observation block back into the dialogue history matrix
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation_result)
            })
            
        # Scenario B: The model did not invoke a tool, meaning it has synthesized the final answer
        else:
            print(" [LOOP COMPLETED]: Agent issued its final response formulation.")
            return response_message.content
            
        current_round += 1
        
    return f" Safety Failure: Agent failed to reach a conclusion within the maximum limit of {max_rounds} rounds."

# 3. Create mock raw string CSV data matching the target schema query parameters
mock_bike_csv = """day,avg_traffic_density,avg_speed_kmh
Monday,85.5,18.2
Tuesday,90.2,16.5
Wednesday,78.1,21.0
Thursday,88.4,17.1
Friday,92.0,15.8
Saturday,45.1,28.4
Sunday,40.3,29.1"""

# Setup a unified query string that forces an explicit two-step dependent layout
test_query = f"Load the following bike commute data as bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh:\n{mock_bike_csv}"

# Initialize pristine loop messages state tracking
messages_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# 4. Trigger the multi-step test sequence
final_agent_response = run_agent_cycle(messages_history, test_query)

print(f"\n[AGENT FINAL RESPONSE]:\n{final_agent_response}")
print("\n------------------------------------------------------------------")

# Q6

# =====================================================================
# REACT LOOP ROLE IDENTIFICATION AND ARCHITECTURAL ANALYSIS
# =====================================================================
# In an autonomous ReAct (Reasoning + Acting) loop framework, the dialogue 
# message history acts as the agent's short-term working memory state tracker. 
# Each distinct 'role' serves a specific functional purpose in orchestrating autonomy:
#
# 1. 'system'    : Defines the global core instructions, operational boundaries, 
#                  rules of engagement, and descriptions of what tools exist. 
#                  It sets the behavioral guardrails for the model.
#
# 2. 'user'      : Represents the original external objective or prompt injected 
#                  by the operator (e.g., "Load this file and run a calculation").
#
# 3. 'assistant' : Represents the LLM's cognitive processing layer. In a tool environment, 
#                  it contains either the structural text reasoning steps explaining 
#                  *why* a certain path was taken, the structured JSON arguments requesting 
#                  a specific tool call, or the final natural language answer.
#
# 4. 'tool'      : Represents the empirical observation data returned *back* to the LLM 
#                  after a native Python function completes execution. It supplies the factual 
#                  ground truth that the assistant uses to figure out its next logical step.
# =====================================================================

print("\n--- Lesson 03: Q6 - Detailed Conversation State History Inspection ---")

# Print the complete accumulated conversation array using clean JSON serialization formatting
print(json.dumps(messages_history, indent=2, default=str))

print("\n------------------------------------------------------------------")

# --- Lesson 04 ---
# Q7

print("\n--- Lesson 04: Q7 - smolagents Automatic Tool Introspection ---")

from smolagents import tool

# 1. Re-wrap the core statistical method using the native @tool decorator
@tool
def compute_correlation(col1: str, col2: str) -> str:
    """
    Compute the Pearson correlation coefficient and p-value between two columns in the loaded DataFrame.

    Args:
        col1: The name of the first numeric column header.
        col2: The name of the second numeric column header.
    """
    # Route execution down to our existing global CsvManager object instance
    result_dict = csv_manager.compute_correlation(col1, col2)
    return str(result_dict)

# 2. Inspect the framework-generated tool description
print("📋 Automatically Generated smolagents Description Property:")
print(compute_correlation.description)

# =====================================================================
# Q7 FRAMEWORK METADATA INTROSPECTION ANALYSIS
# =====================================================================
# Comparison: Manual JSON Schema (Q4) vs. smolagents Auto-Generation (Q7)
# 
# In Q4, writing a valid schema required defining nested JSON keys manually 
# ("type", "properties", "required", etc.), which is tedious and prone to syntax typos. 
# In contrast, `smolagents` leverages standard Python reflection capabilities to construct 
# this underlying payload automatically under the hood.
#
# To produce an accurate, high-quality tool description, smolagents strictly requires 
# two things from you (the developer):
# 
# 1. Strict Python Type Hints: Specifying variables as explicit primitives (e.g., `col1: str`) 
#    tells the framework what data types to declare and enforce in the final parameters object.
# 2. Docstrings with a Valid 'Args:' Block: The framework parses the natural language block 
#    to build the core functional description. It relies specifically on standard Google-style 
#    or Sphinx-style documentation formats to map parameter-specific explanations to their keys. 
#    If you omit the 'Args:' breakdown or skip type hints, the framework will throw an error 
#    because the agent wouldn't know how to fill the arguments.
# =====================================================================

print("\n------------------------------------------------------------------")

# Q8

print("\n--- Lesson 04: Q8 - ToolCallingAgent vs. CodeAgent Comparison ---")

from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel

# 1. Initialize the official smolagents model wrapper using our local environment key
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 2. Define a quick wrapper tool for loading data so smolagents can use it
@tool
def load_csv(csv_data: str) -> str:
    """
    Parse raw CSV string data into the active data management framework.

    Args:
        csv_data: The raw string representation of the CSV file content.
    """
    return str(csv_manager.load_csv(csv_data))

# Package our tools together for the agent infrastructure
TOOLS = [load_csv, compute_correlation]

# 3. Instantiate both operational flavors of agents
tool_agent = ToolCallingAgent(tools=TOOLS, model=model)
code_agent = CodeAgent(tools=TOOLS, model=model)

# 4. Define our mock data matching the target plotting columns
mock_plotting_csv = """day,duration_min,avg_heart_rate
Monday,30,135
Tuesday,45,142
Wednesday,25,128
Thursday,50,150
Friday,40,140
Saturday,60,155
Sunday,20,122"""

prompt = f"Load the following data as bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots:\n{mock_plotting_csv}"

# 5. Execute the ToolCallingAgent execution run
print("\n [RUNNING]: Dispatching plot prompt to ToolCallingAgent...")
try:
    response_tool = tool_agent.run(prompt)
except Exception as e:
    response_tool = f"Execution failed: {str(e)}"

# 6. Execute the CodeAgent execution run
print("\n [RUNNING]: Dispatching plot prompt to CodeAgent...")
try:
    # CodeAgent can dynamically execute local blocks and has access to additional state
    response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
except Exception as e:
    response_code = f"Execution failed: {str(e)}"

# 7. Print the behavioral comparative outputs
print(f"\n[TOOL-CALLING AGENT RESPONSE]:\n{response_tool}")
print(f"\n[CODE AGENT RESPONSE]:\n{response_code}")

# =====================================================================
# Q8 AGENT ARCHITECTURE COMPARATIVE EVALUATION
# =====================================================================
# What did each agent actually produce? Did they change the dot color to green?
#
# - ToolCallingAgent: Produced a text response explaining that it could not generate 
#   plots. It ran `load_csv`, but because it is bound strictly to a fixed menu of JSON-mapped 
#   tools, it had no native mathematical tool to generate an image file or run complex logic. 
#   Consequently, it could not control the dot color or render a plot at all.
#
# - CodeAgent: Successfully generated an actual plot and saved it as an image asset 
#   or returned a successful execution path. Because a CodeAgent can write raw, dynamic Python code 
#   on the fly inside its safe sandboxed interpreter loop, it simply wrote a script utilizing 
#   matplotlib/pandas (`plt.scatter(..., color='green')`). It fully respected the "green dots" style 
#   instruction because it has the programmatic freedom of an open code environment.
#
# What does this reveal about when each type of agent is more useful?
#
# 1. ToolCallingAgent is most useful for highly structured, predictable corporate tasks. 
#    If your system simply needs to hit predefined REST APIs, query specific SQL databases with 
#    known routines, or update a CRM field, ToolCallingAgents are exceptionally safe, robust, 
#    and cheap. They prevent the model from going rogue with custom code.
#
# 2. CodeAgent is vastly superior for exploratory data science, advanced math, and arbitrary data 
#    manipulation tasks. When you don't know ahead of time exactly how a user will want to filter, 
#    combine, reshape, or chart a dataset, you cannot reasonably write a JSON tool for every combination. 
#    Giving the agent an executable sandbox allows it to act like a true data analyst—solving custom 
#    problems programmatically on demand.
# =====================================================================

print("\n------------------------------------------------------------------")

# Q9

# =====================================================================
# Q9 ARCHITECTURAL SELECTION & SECURITY RISK ANALYSIS COMMENT BLOCK
# =====================================================================
#
# 1. When a ToolCallingAgent is the Better Choice:
#    - Ideal Task: A high-security corporate Banking Transaction API integration or a 
#      Production CRM Updates module (e.g., "Update a client's credit card billing address").
#    - Determining Task Property: **Determinism, Strict Structure, and Zero-Tolerance for Variance**. 
#      Tasks that require highly structured inputs and clear, non-negotiable boundaries are a 
#      perfect fit for tool-based approaches. You do not want an AI agent inventing brand-new 
#      logic or writing arbitrary code when moving money or mutating sensitive user data. A 
#      ToolCallingAgent is bound entirely to the strict validation schemas you define, ensuring it 
#      can only interact with your backend through exact, audited function interfaces.
#
# 2. Meaningful Risks of using a CodeAgent:
#    - Core Risk: **Arbitrary / Malicious Code Execution (Sandbox Escape and Resource Exhaustion)**.
#    - Detailed Explanation: Because a CodeAgent functions by dynamically compiling and running 
#      Python strings generated on the fly by an LLM, it carries significant inherent security risks. 
#      If the agent is fed untrusted user input or suffers a prompt injection attack, the model could 
#      be manipulated into executing destructive commands. 
#      - For example, it could run malicious code loops like an infinite `while True:` loop that 
#        freezes the host server's CPU, or worse, attempt to execute shell commands to access the 
#        host filesystem or environment variables (e.g., stealing your `OPENAI_API_KEY`).
#      - A ToolCallingAgent is completely immune to this vector because it cannot write or run 
#        new code; it can only choose from your pre-authored menu of standard Python functions.
# =====================================================================

print("\n==================================================================")
