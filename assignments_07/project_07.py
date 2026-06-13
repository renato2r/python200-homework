import os
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import scipy.stats as stats

# [FIX FOR MATPLOTLIB GUI THREAD ERROR]: Force non-interactive backend to prevent GUI thread rendering crashes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smolagents import CodeAgent, OpenAIServerModel, tool

# 1. Load system environment variables and apply API key assertion guardrails
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "❌ Error: OPENAI_API_KEY not found in .env file!"
print("🚀 [CONFIRMATION]: OpenAI API Key verified and bound for World Happiness Agent.")

#TEST
# 2. Establish and verify runtime outputs folder framework
project_outputs_dir = Path("./outputs")
project_outputs_dir.mkdir(parents=True, exist_ok=True)
print(f"📂 [CONFIRMATION]: Target project outputs directory confirmed at: {project_outputs_dir}")

# 3. Handle data routing dynamically
DATA_PATH = Path("../assignments_01/outputs/merged_happiness.csv")

if DATA_PATH.exists():
    print(f"📊 [DATA LINKED]: Successfully located Week 1 merged data asset at '{DATA_PATH}'")
else:
    print("⚠️ [DATA WARNING]: Week 1 merged_happiness.csv not found at the standard path.")
    print("👉 Code will assume a localized file or handle raw ingestion via Task 1 tools.")
    DATA_PATH = Path("./assignments_07/merged_happiness.csv")

print("\n--- Pre-task: Ingestion and Environment Configuration Complete ---")
print("------------------------------------------------------------------")


# =====================================================================
# --- TASK 1: DEFINE YOUR TOOLS ---------------------------------------
# =====================================================================

print("\n--- Task 1: Initializing Global State and Tools ---")

# Declare the shared global DataFrame that all subsequent tools will mutate or query
df = None

@tool
def load_happiness_data() -> dict:
    """
    Load the World Happiness dataset into the active global memory space.
    If the pre-merged dataset is not found, this tool falls back to dynamically 
    compiling and parsing individual annual records from the resource repository.

    Returns:
        A dictionary containing structural dataset metadata, specifically 'shape' and 'columns'.
    """
    global df
    primary_path = Path("../assignments_01/outputs/merged_happiness.csv")
    fallback_dir = Path("assignments/resources/happiness_project")
    
    print(f"🔄 [TOOL EXECUTION]: load_happiness_data triggered...")
    
    if primary_path.exists():
        print(f"  📥 Found merged dataset at '{primary_path}'. Loading directly...")
        df = pd.read_csv(primary_path)
    elif fallback_dir.exists():
        print(f"  ⚠️ Merged file missing. Scanning fallback directory: '{fallback_dir}'...")
        csv_files = list(fallback_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"❌ Error: No CSV files discovered inside fallback resource path '{fallback_dir}'")
            
        compiled_frames = []
        for file in sorted(csv_files):
            print(f"    Parsing annual file component: {file.name}")
            temp_df = pd.read_csv(file)
            
            rename_map = {
                "Happiness Score": "happiness_score",
                "Life Ladder": "happiness_score",
                "Score": "happiness_score",
                "Country or region": "country",
                "Country name": "country",
                "Year": "year"
            }
            temp_df = temp_df.rename(columns=rename_map)
            
            if "country" in temp_df.columns and "happiness_score" in temp_df.columns:
                compiled_frames.append(temp_df[["country", "year", "happiness_score"]])
        
        if not compiled_frames:
            raise ValueError("❌ Error: Failed to extract compatible data arrays from resource files.")
            
        df = pd.concat(compiled_frames, ignore_index=True)
        print("  ✅ Dynamic multi-year merge operation finished successfully.")
    else:
        local_path = Path("./assignments_07/merged_happiness.csv")
        if local_path.exists():
            print(f"  📥 Found local copies inside workspace at '{local_path}'. Ingesting...")
            df = pd.read_csv(local_path)
        else:
            raise FileNotFoundError("❌ Fatal Data Discovery Failure: Unable to locate data assets.")

    return {
        "status": "success",
        "shape": df.shape,
        "columns": list(df.columns)
    }

@tool
def get_raw_data() -> pd.DataFrame:
    """
    Retrieve the fully loaded pandas DataFrame containing the World Happiness data.
    Use this tool whenever you need to perform direct python coding, plotting, 
    grouping, or advanced analytics that other tools do not cover.

    Returns:
        The active pandas DataFrame object.
    """
    global df
    if df is None:
        # Prevent premature empty state failures by forcing data ingestion
        load_happiness_data()
    return df

@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The exact name of the column header to calculate statistics for.
    """
    global df
    print(f"🔄 [TOOL EXECUTION]: summarize_column triggered for target column: '{column}'")
    
    if df is None:
        return {"error": "No dataset has been loaded into memory yet."}
    if column not in df.columns:
        return {"error": f"Column '{column}' not discovered in dataset.", "available_columns": list(df.columns)}
        
    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The name of the first numeric column header.
        col2: The name of the second numeric column header.
    """
    global df
    print(f"🔄 [TOOL EXECUTION]: compute_correlation triggered for columns: '{col1}' and '{col2}'")
    
    if df is None:
        return {"error": "No dataset has been loaded into memory yet."}
    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found."}
        
    clean_df = df[[col1, col2]].dropna()
    r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r_val), 4),
        "p_value": round(float(p_val), 4)
    }

@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The exact name of the column header to rank countries by.
        year: The target calendar year to filter records by.
        n: The total number of top-performing countries to return. Defaults to 5.
    """
    global df
    print(f"🔄 [TOOL EXECUTION]: get_top_n_countries triggered for column: '{column}', year: {year}")
    
    if df is None:
        return {"error": "No dataset has been loaded into memory yet."}
        
    yearly_df = df[df["year"] == int(year)]
    ranked_df = yearly_df.sort_values(by=column, ascending=False).head(n)
    return {
        "year": year,
        "ranked_metric": column,
        "results": ranked_df[["country", column]].to_dict(orient="records")
    }

print("✅ All tools registered successfully.")


# =====================================================================
# --- TASK 2: BUILD THE AGENT ----------------------------------------
# =====================================================================

print("\n--- Task 2: Instantiating the World Happiness CodeAgent ---")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# [FIX FOR CODE PARSING ERROR]: Enforced strict, non-ambiguous regex instructions to prevent markdown mismatches
SYSTEM_PROMPT = """You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations, and ranking countries.

CRITICAL CODE BLOCK INSTRUCTION:
When you need to write custom Python code for plots or complex math, you MUST format your output exactly as follows:
Thoughts: Your reasoning here
<code>
# Your valid python code here
</code>

ALWAYS call the 'get_raw_data()' tool first inside your code block to fetch the active pandas DataFrame.
Do not include conversational conversational text outside or mixed inside the <code> tags once a tool/code step begins. Be concise."""

agent = CodeAgent(
    # Included get_raw_data to allow the sandbox context to pull data updates seamlessly
    tools=[load_happiness_data, get_raw_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# =====================================================================
# SYSTEM MAIN ENTRY POINT ORCHESTRATION 
# =====================================================================
if __name__ == "__main__":
    print("\n🏁 [SYSTEM INITIALIZATION]: Booting up main execution block...")

    # =====================================================================
    # --- TASK 3: RUN GUIDED QUERIES --------------------------------------
    # =====================================================================
    print("\n--- Task 3: Running Conversational Analysis Pipeline ---")

    # [FIX]: Resolve the absolute path dynamically for Query 5 to eliminate directory nesting errors
    absolute_region_plot_path = str((project_outputs_dir / "happiness_by_region.png").resolve()).replace("\\", "/")

    queries = [
        "Load the happiness data using your tool and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        f"Call get_raw_data() to get the DataFrame. Plot happiness_score over the years as a line chart with one line per region. Save the plot exactly to the absolute path destination: {absolute_region_plot_path}."
    ]

    for i, query in enumerate(queries, start=1):
        print(f"\n==================================================================")
        print(f"❓ [QUERY {i}]: {query}")
        print(f"==================================================================")
        try:
            response = agent.run(query, reset=False)
            print(f"\n💡 [AGENT RESPONSE FOR QUERY {i}]:\n{response}")
        except Exception as e:
            print(f"\n❌ [EXECUTION FAILURE]: Error in Query {i}: {str(e)}")

    # Verify creation of the first plot
    target_plot_path = project_outputs_dir / "happiness_by_region.png"
    if target_plot_path.exists():
        print(f"\n🎨 [VERIFICATION SUCCESS]: Saved to disk at: '{target_plot_path}'")
    else:
        print(f"\n⚠️ [VERIFICATION WARNING]: Could not find 'happiness_by_region.png'")

    # =====================================================================
    # --- TASK 4: YOUR OWN QUESTIONS --------------------------------------
    # =====================================================================
    print("\n--- Task 4: Running Custom Complex Agent Enquiries ---")

    # --- Custom Query 1 ---
    my_query_1 = (
        "Call get_raw_data() to fetch the active DataFrame. Find the single country that experienced the "
        "largest absolute drop in its happiness_score between the years 2015 and 2023. "
        "Print its name and how much its score fell."
    )
    print(f"\n==================================================================")
    print(f"❓ [CUSTOM QUERY 1]: {my_query_1}")
    print(f"==================================================================")
    try:
        response_1 = agent.run(my_query_1, reset=False)
        print(f"\n💡 [AGENT RESPONSE FOR CUSTOM QUERY 1]:\n{response_1}")
    except Exception as e:
        print(f"❌ Custom Query 1 Execution failed: {str(e)}")

    # COMMENT ON CUSTOM QUERY 1:
    # Operational Behavior: This query triggers both tool use and code generation.
    # The agent calls get_raw_data() to access the shared dataframe, then generates a custom
    # pandas snippet to slice, calculate differences between 2015 and 2023, sort, and extract the minimum value.

    # --- Custom Query 2 ---
    # [FIX]: Resolve the absolute path dynamically for Custom Query 2 to target the existing structure directly
    absolute_macro_plot_path = str((project_outputs_dir / "macro_trends_dual_axis.png").resolve()).replace("\\", "/")

    my_query_2 = (
        "Call get_raw_data() to fetch the active DataFrame. Group it by year and calculate the mean for gdp_per_capita and social_support. "
        "Create a dual-axis line chart using matplotlib twinx. "
        f"Save the figure exactly to the absolute path destination: {absolute_macro_plot_path}."
    )
    print(f"\n==================================================================")
    print(f"❓ [CUSTOM QUERY 2]: {my_query_2}")
    print(f"==================================================================")
    try:
        response_2 = agent.run(my_query_2, reset=False)
        print(f"\n💡 [AGENT RESPONSE FOR CUSTOM QUERY 2]:\n{response_2}")
    except Exception as e:
        import traceback
        print(f"❌ Custom Query 2 Execution failed! Details: {str(e)}")
        traceback.print_exc()

    # COMMENT ON CUSTOM QUERY 2:
    # Operational Behavior: This query triggers pure code generation within the sandbox environment.
    # The agent groups by year, computes means, establishes a secondary twin axis with twinx(),
    # customizes colors/labels, and saves the file directly to the assigned path.

    # Verify creation of the second plot
    target_custom_plot = project_outputs_dir / "macro_trends_dual_axis.png"
    if target_custom_plot.exists():
        print(f"\n🎨 [VERIFICATION SUCCESS]: Custom plot saved at: '{target_custom_plot}'")
    else:
        print(f"\n⚠️ [VERIFICATION WARNING]: Could not verify macro_trends_dual_axis.png")

    print("\n==================================================================")
    print("🎉 Week 7 World Happiness Agent Mini-Project Complete!")
    print("==================================================================")


# =====================================================================
# --- TASK 5: REFLECTION ----------------------------------------------
# =====================================================================
#
# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
#    - Behavior: The agent accurately evaluated the p-value returned by our custom 
#      `compute_correlation` tool. Because the tool outputs a rounded float near 0.0000, 
#      the agent explicitly called it out as "statistically significant." It used the p-value 
#      correctly by confirming that a value near zero indicates the observed strong correlation 
#      between gdp_per_capita and happiness_score is highly unlikely to have occurred by random chance. 
#      It naturally applied the standard scientific alpha threshold of α = 0.05 to reject 
#      the null hypothesis.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#    - Behavior: The CodeAgent's ability to seamlessly handle Query 5 and Custom Query 2 was 
#      impressive. Instead of crashing or failing due to missing pre-configured tools, it recognized 
#      it had access to a python execution sandbox, called get_raw_data(), and wrote complex 
#      Matplotlib logic including twinx() and multi-line aggregations completely from scratch.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    - Proposed Tool: `run_ols_regression(dependent_var: str, independent_vars: list) -> dict`
#    - Details: This tool would utilize `statsmodels` or `scikit-learn` to execute an Ordinary 
#      Least Squares (OLS) multiple linear regression, returning R-squared, coefficients, and t-stats.
#    - Questions It Would Answer: It would help answer multi-variable impact queries like: 
#      "When controlling for health and social support, does GDP per capita still have 
#      a significant predictive impact on a country's happiness score?"
#
# =====================================================================