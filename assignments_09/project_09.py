# LINK VIDEO : https://youtu.be/EX2tMLodVAs

import json
import os
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

# --- Setup Constants ---

ACCOUNT_URL = "https://renatoctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# --- Pipeline Functions ---

# --- Step 1: Extract ---

def extract_weather_data(lat: float = 35.2271, lon: float = -80.8431) -> dict:
    """
    Fetches 7 days of hourly temperature and precipitation data for a given 
    location from the Open-Meteo API.
    
    :param lat: Latitude of the target city (defaults to Charlotte, NC).
    :param lon: Longitude of the target city (defaults to Charlotte, NC).
    :return: A dictionary containing the parsed JSON API response.
    """
    # Construct the API endpoint URL using the specified coordinates
    api_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_2m,precipitation&forecast_days=7"
    )
    
    print(f"Extracting weather data from Open-Meteo API for coordinates ({lat}, {lon})...")
    
    # Execute the HTTP GET request
    response = requests.get(api_url)
    
    # Automatically raise an HTTPError if the request failed (e.g., 404, 500)
    response.raise_for_status()
    
    print("Extraction successful!")
    
    # Parse and return the JSON payload as a Python dictionary
    return response.json()

# --- Step 2: Serialize ---

def serialize_weather_data(data: dict) -> bytes:
    """
    Serializes a Python dictionary containing weather data into JSON-formatted bytes.
    
    :param data: The weather data dictionary from the API.
    :return: A bytes object containing the UTF-8 encoded JSON data.
    """
    print("Serializing weather data to JSON bytes...")
    
    # Convert the Python dictionary into a compact JSON-formatted string
    json_string = json.dumps(data)
    
    # Encode the string into raw UTF-8 bytes for storage transport
    json_bytes = json_string.encode("utf-8")
    
    print(f"Serialization successful! Total size: {len(json_bytes)} bytes.")
    
    return json_bytes

from datetime import date

# --- Step 3: Load ---

def load_weather_data(container_client: ContainerClient, serialized_data: bytes) -> str:
    """
    Uploads serialized JSON bytes to Azure Blob Storage under a date-partitioned path.
    
    :param container_client: An instantiated Azure ContainerClient object.
    :param serialized_data: The UTF-8 encoded JSON bytes to upload.
    :return: The string representing the destination blob path.
    """
    # Generate today's date in ISO format (YYYY-MM-DD)
    today_str = date.today().isoformat()
    
    # Construct the dynamic partition path
    blob_path = f"raw/{today_str}/weather.json"
    
    print(f"Uploading data to Azure Blob Storage at path: '{blob_path}'...")
    
    # Upload the binary payload, ensuring existing files for today are overwritten
    container_client.upload_blob(name=blob_path, data=serialized_data, overwrite=True)
    
    # Print the mandatory confirmation message
    print(f"SUCCESS: Uploaded {len(serialized_data)} bytes to blob path '{blob_path}'.")
    
    return blob_path

# --- Step 4: Verify ---

def verify_pipeline_load(container_client: ContainerClient) -> None:
    """
    Lists all blobs inside the container to verify the pipeline's 
    output and prints their names and sizes.
    
    :param container_client: An instantiated Azure ContainerClient object.
    :return: None
    """
    print("Verifying container contents...")
    
    # Counter to track if the container is empty
    blob_count = 0
    
    for blob in container_client.list_blobs():
        blob_count += 1
        print(f" -> Found Blob: {blob.name} ({blob.size} bytes)")
        
    if blob_count == 0:
        print("WARNING: The container is currently empty.")
    else:
        print(f"Verification complete. Total blobs found: {blob_count}")

import pandas as pd

# --- Step 5: Read Back ---

def read_back_and_export(container_client: ContainerClient, blob_path: str) -> None:
    """
    Downloads the uploaded JSON blob, parses its hourly weather metrics into 
    a pandas DataFrame, prints a sample, and saves the raw JSON locally.
    
    :param container_client: An instantiated Azure ContainerClient object.
    :param blob_path: The path of the blob to download.
    :return: None
    """
    print(f"Downloading blob from path: '{blob_path}' for analysis...")
    
    # 1. Download the blob content as bytes
    blob_client = container_client.get_blob_client(blob_path)
    download_stream = blob_client.download_blob()
    blob_bytes = download_stream.readall()
    
    # 2. Parse the JSON bytes back into a Python dictionary
    weather_json = json.loads(blob_bytes.decode('utf-8'))
    
    # 3. Load the "hourly" field into a pandas DataFrame
    hourly_data = weather_json.get("hourly", {})
    df = pd.DataFrame(hourly_data)
    
    print("\n--- Pandas DataFrame (First 5 Rows) ---")
    print(df.head())
    print("---------------------------------------\n")
    
    # 4. Save the downloaded JSON to outputs/weather_raw.json
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    output_file_path = os.path.join(output_dir, "weather_raw.json")
    
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(weather_json, f, indent=4)
        
    print(f"SUCCESS: Raw JSON successfully saved locally to '{output_file_path}'.")

# --- Pipeline Orchestration ---

if __name__ == "__main__":
    print("=== STARTING WEATHER DATA PIPELINE ===")
    
    # 1. Inicializa o cliente do Azure Blob Storage usando autenticação automática
    print("\n[Setup] Initializing Azure credentials...")
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=credential)
    container_client = blob_service_client.get_container_client(CONTAINER)
    
    try:
        # STEP 1: Extract
        print("\n[Step 1] Executing Extraction...")
        raw_data = extract_weather_data()
        
        # STEP 2: Serialize
        print("\n[Step 2] Executing Serialization...")
        serialized_bytes = serialize_weather_data(raw_data)
        
        # STEP 3: Load
        print("\n[Step 3] Executing Load to Cloud...")
        uploaded_path = load_weather_data(container_client, serialized_bytes)
        
        # STEP 4: Verify
        print("\n[Step 4] Executing Post-Load Verification...")
        verify_pipeline_load(container_client)
        
        # STEP 5: Read Back
        print("\n[Step 5] Executing Read Back and Analytical Prep...")
        read_back_and_export(container_client, uploaded_path)
        
        print("\n=== PIPELINE EXECUTED SUCCESSFULLY ===")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during execution: {e}")