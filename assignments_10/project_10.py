# VIDEO LINK : https://youtu.be/DtuW4-3oGVU

# ====================================================================
# STEP 6: PIPELINE ARCHITECTURE REFLECTION
# ====================================================================
# Classifying weather conditions for outdoor running is a poor use of an LLM, 
# as the task relies entirely on quantitative bounds that deterministic code handles 
# infinitely better. Switching to a rule-based approach causes you to lose the model's 
# semantic flexibility to interpret colloquial or unstructured data inputs, but you gain 
# 100% predictable accuracy, near-instant execution speed, and zero API execution costs. 
# In a production data pipeline, deterministic logic should always be favored for explicit, 
# numeric decision boundaries, leaving LLMs reserved for non-structured text transformations.
# ====================================================================

import os
import json
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
ACCOUNT_URL = "https://renatoctd2026sa.blob.core.windows.net"  # Replace with your real storage account name
CONTAINER = "pipeline-data"
TODAY_STR = date.today().isoformat()
INPUT_BLOB_PATH = f"raw/{TODAY_STR}/weather.json"
OUTPUT_BLOB_PATH = f"processed/{TODAY_STR}/weather_classified.json"


# --- Step 1: Read From Blob Storage ---

def download_raw_weather(container_client: ContainerClient, blob_path: str) -> list[dict]:
    """
    Downloads the raw weather JSON data from Azure Blob Storage, or falls back 
    to a local resource file if the blob is missing. Reshapes parallel lists 
    into a list of per-hour record dictionaries.
    
    :param container_client: Instantiated Azure ContainerClient object.
    :param blob_path: Target cloud path for the source weather data.
    :return: A list of dictionaries, where each dictionary represents one hour of data.
    """
    raw_data = None
    
    try:
        print(f"Attempting to download raw data from Azure blob path: '{blob_path}'...")
        blob_client = container_client.get_blob_client(blob_path)
        download_stream = blob_client.download_blob()
        blob_bytes = download_stream.readall()
        raw_data = json.loads(blob_bytes.decode("utf-8"))
        print("Successfully retrieved weather data from Azure Blob Storage.")
        
    except Exception as e:
        print(f"Blob Storage retrieval failed or file missing: {e}")
        fallback_path = "assignments/resources/weather_raw.json"
        print(f"Activating Fallback Mechanism: Loading local file from '{fallback_path}'...")
        
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(f"Both Azure Blob and local fallback data at '{fallback_path}' are missing.")
            
        with open(fallback_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    hourly_raw = raw_data.get("hourly", {})
    if not hourly_raw or "time" not in hourly_raw:
        raise KeyError("Input data structure is missing the required 'hourly' parallel lists.")
        
    reshaped_records = []
    total_records = len(hourly_raw["time"])
    
    for i in range(total_records):
        record = {
            "time": hourly_raw["time"][i],
            "temperature_2m": hourly_raw["temperature_2m"][i],
            "precipitation": hourly_raw["precipitation"][i]
        }
        reshaped_records.append(record)
        
    print(f"Reshaping completed. Processed {len(reshaped_records)} sequential hourly records.")
    return reshaped_records


# --- Step 2: LLM Enrichment Step ---

def enrich_weather_with_llm(openai_client: OpenAI, reshaped_records: list[dict]) -> list[dict]:
    """
    Processes the first 24 weather records through the OpenAI API to classify running conditions.
    Implements validation logic to handle unexpected model outputs gracefully.
    
    :param openai_client: Instantiated OpenAI API client object.
    :param reshaped_records: List of record dictionaries from Step 1.
    :return: A new list containing the first 24 records enriched with a 'conditions' key.
    """
    print("Initiating LLM Classification for outdoor running conditions...")
    
    # Exact system prompt requested by the mentor guidelines
    SYSTEM_PROMPT = (
        "You are classifying hourly weather conditions for outdoor running. "
        "Given a temperature in Celsius and a precipitation amount in mm, "
        "classify the conditions as exactly one of: good, marginal, or bad. "
        "Reply with that one word only -- no punctuation, no explanation."
    )
    
    target_records = reshaped_records[:24]
    enriched_records = []
    valid_labels = {"good", "marginal", "bad"}
    
    for index, record in enumerate(target_records):
        user_message = f"Temperature: {record['temperature_2m']}C, Precipitation: {record['precipitation']}mm"
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":user_message}
                ],
                temperature=0.0
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            if classification not in valid_labels:
                print(f" [Warning] Unexpected LLM output received: '{classification}'. Defaulting to 'unknown'.")
                classification = "unknown"
                
        except Exception as e:
            print(f" [Error] API call failed at record index {index}: {e}")
            classification = "unknown"
            
        enriched_record = record.copy()
        # Changed key from 'running_condition' to 'conditions' as requested by mentor
        enriched_record["conditions"] = classification
        enriched_records.append(enriched_record)
        
        if (index + 1) % 6 == 0:
            print(f" -> Progress Update: Processed {index + 1}/24 records successfully.")
            
    print(f"LLM Transformation complete. Generated {len(enriched_records)} enriched records.")
    return enriched_records


# --- Step 3: Load Enriched Data back to Blob ---

def upload_processed_data(container_client: ContainerClient, blob_path: str, records: list[dict]) -> None:
    """
    Serializes the enriched record data list into a JSON string and uploads 
    it back to Azure Blob Storage, overwriting any pre-existing file.
    
    :param container_client: Instantiated Azure ContainerClient object.
    :param blob_path: Destination cloud path for the processed blob.
    :param records: List of enriched record dictionaries to store.
    :return: None
    """
    print(f"Uploading enriched dataset to processed storage path: '{blob_path}'...")
    
    try:
        blob_client = container_client.get_blob_client(blob_path)
        serialized_json_bytes = json.dumps(records, indent=4).encode("utf-8")
        blob_client.upload_blob(serialized_json_bytes, overwrite=True)
        print("SUCCESS: Enriched file successfully written to Azure Blob Storage.")
        
    except Exception as e:
        print(f"[Error] Failed to upload processed data to Blob Storage: {e}")
        raise e


# --- Step 4: Analytical Spot-Check ---

def spot_check_processed_data(container_client: ContainerClient, blob_path: str) -> None:
    """
    Downloads the processed JSON dataset from Blob Storage, loads it into a 
    pandas DataFrame, and prints summary metrics along with a data sample.
    
    :param container_client: Instantiated Azure ContainerClient object.
    :param blob_path: Cloud path of the processed blob to verify.
    :return: None
    """
    print(f"\nDownloading processed blob from '{blob_path}' for final verification...")
    
    blob_client = container_client.get_blob_client(blob_path)
    download_stream = blob_client.download_blob()
    processed_bytes = download_stream.readall()
    
    records = json.loads(processed_bytes.decode("utf-8"))
    df = pd.DataFrame(records)
    
    print("\n" + "="*50)
    print("         FINAL PIPELINE SPOT-CHECK METRICS")
    print("="*50)
    
    # Updated column references to target "conditions"
    print("\n--- Value Counts for Running Conditions ---")
    if "conditions" in df.columns:
        print(df["conditions"].value_counts())
    else:
        print("[Warning] 'conditions' column not found in DataFrame.")
        
    print("\n--- First 5 Rows of Processed DataFrame ---")
    print(df.head(5))
    print("="*50 + "\n")


# --- Main Pipeline Orchestrator ---

if __name__ == "__main__":
    print("=== STARTING LLM TRANSFORM WEATHER PIPELINE ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Critical Error: Missing 'OPENAI_API_KEY' inside environment configuration.")
        
    openai_client = OpenAI()
    
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    
    try:
        # STEP 1: Download and reshape parallel source structures
        print("\n--- PHASE 1: DATA INGESTION & RESHAPING ---")
        raw_records = download_raw_weather(container_client, INPUT_BLOB_PATH)
        
        # STEP 2: Transform subset records with deterministic LLM evaluation
        print("\n--- PHASE 2: LLM CONDITION CLASSIFICATION ---")
        enriched_results = enrich_weather_with_llm(openai_client, raw_records)
        
        # STEP 3: Persist structured telemetry back to cloud Data Lake
        print("\n--- PHASE 3: PRODUCTION CLOUD LOAD ---")
        upload_processed_data(container_client, OUTPUT_BLOB_PATH, enriched_results)
        
        # STEP 4: Download back and execute final spot check
        print("\n--- PHASE 4: ANALYTICAL SPOT-CHECK ---")
        spot_check_processed_data(container_client, OUTPUT_BLOB_PATH)
        
        # STEP 5: Export complete structural array locally for mentor auditing
        print("\n--- PHASE 5: EXPORTING SAMPLE ARTIFACTS ---")
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        local_mentor_file = os.path.join(output_dir, "first_10_records.json")
        
        # Slice only the first 10 records for your mentor
        mentor_sample_slice = enriched_results[:10]
        
        with open(local_mentor_file, "w", encoding="utf-8") as f:
            json.dump(mentor_sample_slice, f, indent=4)
            
        print(f"SUCCESS: First 10 enriched records successfully saved locally to '{local_mentor_file}'.")
        print("\n=== PIPELINE RUN EXECUTED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Data Pipeline stopped abruptly: {e}")