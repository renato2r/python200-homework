# VIDEO LINK https://youtu.be/usv6gtBmvuc

# assignments_11/etl_pipeline.py

"""
====================================================================
Week 11 Capstone: Orchestrated Cloud ETL Pipeline
====================================================================
"""

import os
import json
from datetime import date
import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import task, flow, get_run_logger
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

# Load environment infrastructure variables from local configuration
load_dotenv()

# --- Global Configurations & Infrastructure Constants ---
ACCOUNT_URL = "https://renatoctd2026sa.blob.core.windows.net"  # Replace with your real storage account name
CONTAINER_NAME = "pipeline-data"
TODAY_STR = date.today().isoformat()
OUTPUT_BLOB_PATH = f"final/{TODAY_STR}/weather_etl.json"

# Geographical Coordinates Selection: San Francisco, California
LATITUDE = 37.7749
LONGITUDE = -122.4194


# --- Step 1: Extract Operations ---

@task(
    name="extract_weather_api",
    retries=2,
    retry_delay_seconds=10
)
def extract_weather_data() -> dict:
    """
    Queries the Open-Meteo REST API to extract 7 days of raw hourly telemetry.
    Configured with a 2-tier automated retry structure to guarantee network resilience.
    """
    logger = get_run_logger()
    logger.info("Initializing Extract Phase: Connecting to Open-Meteo API backend...")
    
    api_url = "https://api.open-meteo.com/v1/forecast"
    parameters = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["temperature_2m", "precipitation"],
        "forecast_days": 7
    }
    
    response = requests.get(api_url, params=parameters, timeout=15)
    
    # Enforce strict error propagation for transient server or client faults
    response.raise_for_status()
    
    raw_json_payload = response.json()
    logger.info("SUCCESS: Data Extraction complete. Raw payload successfully parsed to memory.")
    print("CONFIRMATION: Weather telemetry extracted cleanly from upstream REST interface.")
    
    return raw_json_payload


# --- Step 2: Transform Operations ---

@task(name="transform_and_enrich_llm")
def transform_weather_data(raw_data: dict) -> list[dict]:
    """..."""
    logger = get_run_logger()
    openai_client = OpenAI() # <--- Adicione isto aqui dentro!
    logger.info("Initializing Transform Phase...")
    
    hourly_raw = raw_data.get("hourly", {})
    if not hourly_raw or "time" not in hourly_raw:
        raise KeyError("Extraction payload schema error: 'hourly' data block missing.")
        
    reshaped_records = []
    total_elements = len(hourly_raw["time"])
    
    # Reshape parallel array telemetry arrays into structural array dictionaries
    for i in range(total_elements):
        record = {
            "time": hourly_raw["time"][i],
            "temperature_2m": hourly_raw["temperature_2m"][i],
            "precipitation": hourly_raw["precipitation"][i]
        }
        reshaped_records.append(record)
        
    logger.info(f"Reshaping complete. Formatted {len(reshaped_records)} rows. Slicing first 24 slots for evaluation.")
    
    # Slicing the first 24 records (exactly one full weather day)
    evaluation_subset = reshaped_records[:24]
    enriched_output_list = []
    valid_grading_labels = {"good", "marginal", "bad"}
    
    SYSTEM_PROMPT = (
        "You are classifying hourly weather conditions for outdoor running. "
        "Given a temperature in Celsius and a precipitation amount in mm, "
        "classify the conditions as exactly one of: good, marginal, or bad. "
        "Reply with that one word only -- no punctuation, no explanation."
    )
    
    for index, item in enumerate(evaluation_subset):
        user_input_string = f"Temperature: {item['temperature_2m']}C, Precipitation: {item['precipitation']}mm"
        
        try:
            chat_completion = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input_string}
                ],
                temperature=0.0
            )
            
            model_label = chat_completion.choices[0].message.content.strip().lower()
            
            # Enforce validation gate metrics against unexpected responses
            if model_label not in valid_grading_labels:
                logger.warning(f"Unexpected token string received: '{model_label}'. Activating safety fallback.")
                model_label = "unknown"
                
        except Exception as exc:
            logger.error(f"Inference pipeline execution error at index {index}: {exc}")
            model_label = "unknown"
            
        enriched_item = item.copy()
        enriched_item["conditions"] = model_label  # Matched exact key schema name requested
        enriched_output_list.append(enriched_item)
        
        # Periodic runtime validation logging requirement
        if (index + 1) % 6 == 0:
            print(f"PROGRESS METRIC: Processed and transformed {index + 1}/24 sequential rows.")
            
    logger.info(f"SUCCESS: Structural transformation complete. Enriched array size: {len(enriched_output_list)}")
    return enriched_output_list


# --- Step 3: Load Operations ---

@task(name="load_processed_data_blob")
def load_processed_data(container_client: ContainerClient, blob_destination_path: str, records_list: list[dict]) -> None:
    """
    Serializes data frames to string elements and updates the cloud destination repository,
    enforcing deterministic overwrite boundaries.
    """
    logger = get_run_logger()
    logger.info(f"Initializing Load Phase: Storing structural JSON payload into target path: '{blob_destination_path}'...")
    
    try:
        blob_target_client = container_client.get_blob_client(blob_destination_path)
        
        # Format dataset content with 4-space structural indents
        serialized_data_bytes = json.dumps(records_list, indent=4).encode("utf-8")
        byte_dimension_length = len(serialized_data_bytes)
        
        # Enforce idempotent atomic overwrite operations
        blob_target_client.upload_blob(serialized_data_bytes, overwrite=True)
        
        logger.info("SUCCESS: Production cloud storage load transaction resolved successfully.")
        print(f"CONFIRMATION: Upload complete. Destination Path: '{blob_destination_path}' | Size: {byte_dimension_length} bytes.")
        
    except Exception as exc:
        logger.critical(f"Data ingestion engine halted unexpectedly during payload load: {exc}")
        raise exc


# --- Core Flow Orchestration Entry Point ---

@flow(name="Cloud-ETL-Weather-Capstone-Pipeline", log_prints=True)
def run_weather_etl_pipeline() -> None:
    """
    The parent execution graph orchestrating Extract, Transform, and Load 
    tasks into a highly unified and visible production operation block.
    """
    logger = get_run_logger()
    logger.info("=== INITIALIZING END-TO-END CAPSTONE ETL EXECUTION CORE ===")
    
    # Pre-flight environment check
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Execution Halted: Missing required 'OPENAI_API_KEY' inside environmental profile context.")
        
    # Instantiate API client objects
    openai_engine_client = OpenAI()
    
    # Establish connection mapping patterns for storage resource
    #azure_identity_credentials = DefaultAzureCredential()
    #storage_service_client = BlobServiceClient(ACCOUNT_URL, credential=azure_identity_credentials)
    #target_container_client = storage_service_client.get_container_client(CONTAINER_NAME)
    
    # Establish connection using local environment Connection String
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise EnvironmentError("Missing 'AZURE_STORAGE_CONNECTION_STRING' in environment configuration.")
        
    storage_service_client = BlobServiceClient.from_connection_string(connection_string)
    target_container_client = storage_service_client.get_container_client(CONTAINER_NAME)
    
    # Phase 1: Extract Pipeline Processing Execution
    raw_weather_payload = extract_weather_data()
    
    # Phase 2: Transform Pipeline Processing Execution
    enriched_weather_results = transform_weather_data(raw_weather_payload)
    
    # Phase 3: Load Pipeline Processing Execution
    load_processed_data(target_container_client, OUTPUT_BLOB_PATH, enriched_weather_results)
    
    logger.info("=== DEPLOYMENT WORKFLOW PIPELINE RUN CONCLUDED EXCELLENTLY ===")
    print(f"COMPLETION LOG SUMMARY: Clean data lake records committed directly to storage at: '{OUTPUT_BLOB_PATH}'")


if __name__ == "__main__":
    # Allows for simple localized terminal execution or execution tracking profiling tests
    run_weather_etl_pipeline()