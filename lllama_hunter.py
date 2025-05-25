import shodan
import aiohttp
import asyncio
import json
import os
import argparse # Added argparse
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import login
from datetime import datetime
import time
from getpass import getpass
import nest_asyncio

# Handle nested event loops in Jupyter/Colab
nest_asyncio.apply()

# Securely get API keys
# Prioritize environment variables, then fall back to getpass
SHODAN_API_KEY = os.getenv("SHODAN_API_KEY")
if not SHODAN_API_KEY:
    print("Shodan API key not found in environment variable SHODAN_API_KEY.")
    SHODAN_API_KEY = getpass("Please enter your Shodan API key: ")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("HuggingFace token not found in environment variable HF_TOKEN.")
    HF_TOKEN = getpass("Please enter your HuggingFace API token: ")

# Log into Hugging Face
try:
    login(token=HF_TOKEN)
    print("Successfully logged into Hugging Face Hub.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    # Decide if the script should exit or continue if login fails.
    # For now, it will continue and fail later if push_to_hub is attempted.

# Define dataset structure (globally, as it's a schema)
features = Features({
    "IP": Value("string"),
    "Model": Value("string"),
    "Product": Value("string"),
    "Status": Value("string"),
    "ISP": Value("string"),
    "ASN": Value("string"),
    "Org": Value("string"),
    "Country": Value("string"),
    "City": Value("string"),
    "Longitude": Value("float32"),
    "Latitude": Value("float32"),
    "Port": Value("int32"),
    "Timestamp": Value("string"),
    "Tags": Value("string"),
    "Transport": Value("string"),
})

# Define Shodan fetch function
def fetch_shodan_matches(api, max_matches):
    matches, page = [], 1
    while len(matches) < max_matches:
        try:
            results = api.search("product:Ollama port:11434", page=page)
            matches.extend(results['matches'][:max_matches - len(matches)])
            page += 1
            time.sleep(1)  # Avoid rate limits
        except shodan.APIError as e:
            print(f"[{datetime.now().isoformat()}] Shodan error: {e}")
            break
    return matches

# Async function to query Ollama instances
async def query_ollama(match, session):
    ip = match['ip_str']
    url = f"http://{ip}:11434/api/tags"
    isp, asn, org = match.get('isp', ''), match.get('asn', ''), match.get('org', '')
    location = match.get('location', {})
    country, city = location.get('country_name', ''), location.get('city', '')
    longitude, latitude = location.get('longitude', 0.0), location.get('latitude', 0.0)
    port, tags, product, transport = match['port'], ", ".join(match.get('tags', [])), match.get('product', 'Ollama'), "TCP"

    timestamp = datetime.now().isoformat()

    # Accessing retry_limit and timeout_seconds from the main_async function's scope
    # This requires them to be passed or accessed if main_async is refactored to take args
    # For now, assuming they are accessible if this function is nested or they are passed down.
    # If query_ollama is called from main_async(args), then args.retry_limit, args.timeout_seconds

    # Let's assume main_async will pass these:
    # async def query_ollama(match, session, current_retry_limit, current_timeout_seconds):
    # For the diff, we'll just use RETRY_LIMIT and TIMEOUT_SECONDS as placeholders
    # for where args.retry_limit and args.timeout_seconds would be used.
    # This part of the diff will be more illustrative for the logic change.

    for attempt in range(RETRY_LIMIT): # This would become args.retry_limit
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS) as response: # This would become args.timeout_seconds
                status = str(response.status)
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    return [{
                        "IP": ip, "Model": model.get('name', 'N/A'), "Product": product, "Status": status,
                        "ISP": isp, "ASN": asn, "Org": org, "Country": country, "City": city,
                        "Longitude": longitude, "Latitude": latitude, "Port": port, "Timestamp": timestamp,
                        "Tags": tags, "Transport": transport
                    } for model in models] or [{
                        "IP": ip, "Model": "N/A", "Product": product, "Status": "200 but no models",
                        "ISP": isp, "ASN": asn, "Org": org, "Country": country, "City": city,
                        "Longitude": longitude, "Latitude": latitude, "Port": port, "Timestamp": timestamp,
                        "Tags": tags, "Transport": transport
                    }]
                else:
                    return [{
                        "IP": ip, "Model": "N/A", "Product": product, "Status": status,
                        "ISP": isp, "ASN": asn, "Org": org, "Country": country, "City": city,
                        "Longitude": longitude, "Latitude": latitude, "Port": port, "Timestamp": timestamp,
                        "Tags": tags, "Transport": transport
                    }]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"[{datetime.now().isoformat()}] Attempt {attempt+1}/{RETRY_LIMIT} failed for {ip}: {e}") # This would become args.retry_limit
            await asyncio.sleep(2)  # Short delay before retrying
    return [{
        "IP": ip, "Model": "N/A", "Product": product, "Status": "Failed after retries",
        "ISP": isp, "ASN": asn, "Org": org, "Country": country, "City": city,
        "Longitude": longitude, "Latitude": latitude, "Port": port, "Timestamp": timestamp,
        "Tags": tags, "Transport": transport
    }]

# Define and run main_async function
async def main_async(args):
    # Use args from command line
    global RETRY_LIMIT, TIMEOUT_SECONDS # Allow modification for query_ollama if not passed directly
    RETRY_LIMIT = args.retry_limit
    TIMEOUT_SECONDS = args.timeout_seconds

    api = shodan.Shodan(SHODAN_API_KEY) # SHODAN_API_KEY is still global from getpass/env
    matches = fetch_shodan_matches(api, args.max_matches)
    print(f"[{datetime.now().isoformat()}] Fetched {len(matches)} matches from Shodan using max_matches={args.max_matches}")
    
    async with aiohttp.ClientSession() as session:
        # Consider passing args.retry_limit and args.timeout_seconds to query_ollama if it's refactored
        tasks = [query_ollama(match, session) for match in matches]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"[{datetime.now().isoformat()}] Async task was cancelled! Retrying...")
            # If main_async is to be retried, it needs args
            return await main_async(args) 

        new_data = [entry for sublist in results for entry in sublist if isinstance(sublist, list)]

    temp_file_name = "temp_data.json" # Could also be an arg
    with open(temp_file_name, "w") as f:
        json.dump(new_data, f)
    print(f"[{datetime.now().isoformat()}] Saved {len(new_data)} entries to {temp_file_name}")

    try:
        existing_dataset = load_dataset(args.repo_name)
        existing_data = existing_dataset["train"].to_list() if "train" in existing_dataset else []
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Dataset '{args.repo_name}' not found: {e}")
        existing_data = []

    dataset = Dataset.from_list(existing_data + new_data, features=features) # 'features' is global
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(args.repo_name)
    print(f"[{datetime.now().isoformat()}] Dataset updated with {len(new_data)} new entries at: https://huggingface.co/datasets/{args.repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama Hunter: Scan for Ollama instances and upload data to Hugging Face Hub.")
    parser.add_argument("--repo_name", type=str, default="latterworks/parsed_ollama_data", help="Hugging Face repository name (e.g., your-username/your-dataset-name).")
    parser.add_argument("--max_matches", type=int, default=15, help="Maximum number of matches to fetch from Shodan.")
    parser.add_argument("--timeout_seconds", type=int, default=8, help="Timeout in seconds for querying Ollama instances.")
    parser.add_argument("--retry_limit", type=int, default=3, help="Retry limit for querying Ollama instances.")
    
    args = parser.parse_args()

    # Need to handle global RETRY_LIMIT and TIMEOUT_SECONDS if query_ollama is not refactored to take them as args
    # This assignment will make them available to query_ollama as it currently stands
    RETRY_LIMIT = args.retry_limit
    TIMEOUT_SECONDS = args.timeout_seconds

    try:
        asyncio.run(main_async(args))  # Works in normal Python environments
    except RuntimeError:
        # Fix for Jupyter/Colab's nested event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main_async(args))
