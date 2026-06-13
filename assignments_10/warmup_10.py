# assignments_10/warmup_10.py

# --- LLMs as Transform ---
# Q1

# a) Parse the string "Jan 5th, 2024" and convert it to an ISO date format like "2024-01-05":
# Response: I would use deterministic code (such as Python's `datetime` or `dateutil` libraries) because date parsing follows well-defined logical patterns that can be computed with 100% accuracy and zero risk of hallucination.

# b) Classify a customer support ticket -- "my card was charged twice" -- into billing, tech support, or general:
# Response: I would use an LLM because classifying intent and semantic context from freeform natural language is a complex text-processing task that cannot be reliably solved using rigid string-matching or deterministic rules.

# c) Calculate the average of a list of numbers:
# Response: I would use deterministic code (such as native Python arithmetic or `numpy`) because mathematical calculations require absolute precision and are executed instantly at zero token cost, whereas LLMs are unreliable and inefficient for arithmetic operations.

# d) Extract the company name from a freeform job title like "Sr. Data Eng @ Acme Corp (contract)":
# Response: I would use an LLM because job titles lack a uniform structure (alternating between "@", "at", separators, or parentheses), making context-aware semantic extraction an ideal task for a language model's flexibility.

# e) Check if a product review has more than 100 words:
# Response: I would use deterministic code (such as Python's `.split()` method combined with `len()`) because counting tokens or words is a simple, exact, and instantaneous algorithmic operation, making an LLM unnecessarily expensive and prone to counting errors.


# Q2

# Downstream Pipeline Problem:
# The original prompt produces freeform, unpredictable natural language text. In a data pipeline, 
# downstream steps (like Pandas transformations or SQL insertion) expect consistent schemas, regular data types, 
# and reliable formats. Conversational summaries will break automated parsing logic and fail silently or crash the pipeline.

# Solution:
# Rewrite the prompt to enforce a structured JSON schema, ensuring predictable output formatting.

system_prompt_revised = """
You are a precise data transformation step in an automated pipeline. 
Analyze the provided product review and extract the required information into a valid JSON object. 
Do not include any conversational filler, markdown formatting (such as ```json), or markdown blocks.

The output must conform exactly to this JSON schema:
{
    "summary": "A concise summary of the review in exactly two sentences.",
    "sentiment": "One of the following string values: POSITIVE, NEGATIVE, or NEUTRAL.",
    "is_complaint": true or false (boolean flag indicating a service grievance)
}
"""

# Q3

# Performance and Throughput Analysis at Scale:
#
# a) Calculation for sequential processing time:
#    Total time = 50,000 records * 1 second/record = 50,000 seconds.
#    Converting to hours: 50,000 / 3,600 ≈ 13.89 hours.
#    Sequential processing would take approximately 13.9 hours to complete.
#
# b) Practical strategies to handle this more efficiently at scale:
#    1. Concurrent / Parallel Requests: Use asynchronous programming (`asyncio` and `aiohttp`/`openai` async client) 
#       or thread pools (`concurrent.futures.ThreadPoolExecutor`) to dispatch hundreds of requests in parallel, 
#       drastically shortening execution time by maximizing the API's Rate Limits (RPM/TPM).
#    2. OpenAI Batch API: Submit requests asynchronously in a single batch file. OpenAI processes the jobs 
#       offline within 24 hours at a 50% cost discount, which is perfect for non-real-time ETL pipelines.

# --- Azure OpenAI ---
# Q1

# Two key reasons an organization might use Azure OpenAI instead of the public OpenAI API:
#
# 1. Enterprise-Grade Security and Data Privacy (RBAC & VNet):
#    Azure OpenAI ensures that customer data (prompts and completions) remains strictly within the 
#    organization's secure Azure tenant boundary. It supports Azure Virtual Networks (VNets), Private Endpoints, 
#    and Role-Based Access Control (RBAC) via Microsoft Entra ID. Crucially, Microsoft guarantees that enterprise 
#    data is never used to train or improve public OpenAI base models.
#
# 2. Unified Infrastructure and SLA Guarantees:
#    It allows enterprises to consolidate their AI workloads under existing Microsoft Enterprise Agreements (EA). 
#    This provides consolidated cloud billing, predictable enterprise uptime Service Level Agreements (SLAs), 
#    and dedicated throughput capacity (Provisioned Throughput Units or PTUs), which prevents regional rate-limiting 
#    issues often encountered on the public API during peak production hours.

# Q2

# The three Azure-specific parameters required to initialize the AzureOpenAI client are:
#
# 1. azure_endpoint (string):
#    The base URL pointing to your deployed Azure OpenAI resource instance in the Azure Portal 
#    (e.g., "https://your-resource-name.openai.azure.com/"). It replaces the public global OpenAI endpoint 
#    and routes requests directly to your dedicated cloud boundary.
#
# 2. api_version (string):
#    The explicit REST API version date that defines the contract for your requests and responses 
#    (e.g., "2024-06-01" or a similar stable release version). Unlike the public OpenAI API which changes 
#    implicitly, Azure requires an explicit version string to ensure predictable, version-locked API behavior.
#
# 3. azure_deployment (string) [or 'model/deployment' mapping inside the call]:
#    The custom, unique name you gave to your deployed model instance within the Azure OpenAI Studio 
#    (e.g., "gpt-4o-mini-prod-deployment"). In Azure, you do not invoke a raw base model name directly; 
#    instead, you target a specific managed deployment instance hosting that model.

# Q3

# What it takes instead:
# In `chat.completions.create()` using AzureOpenAI, the `model` parameter does not take a generic model name. 
# Instead, it takes your custom Azure Deployment Name (e.g., "weather-pipeline-gpt4o-mini"). This string 
# acts as a direct routing identifier to your specific provisioned capacity instance in the cloud.
#
# Where to find the right value:
# 1. Log into the Azure Portal and navigate to your Azure OpenAI Service resource.
# 2. Click on "Go to Azure AI Studio" or "Azure OpenAI Studio".
# 3. Under the "Management" section in the left sidebar, click on "Deployments".
# 4. Find the deployment you want to use from the list, and copy the exact name found under the 
#    "Deployment name" column. This is the exact string needed for the code parameter.