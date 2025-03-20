import shodan
import aiohttp
import asyncio
import json
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import login
from datetime import datetime
import time
from getpass import getpass
import nest_asyncio

# Handle nested event loops in Jupyter/Colab
nest_asyncio.apply()

# Securely get API keys
SHODAN_API_KEY = getpass("Enter your Shodan API key: ")
HF_TOKEN = getpass("Enter your Hugging Face token: ")

# Log into Hugging Face
login(token=HF_TOKEN)

# Set constants
REPO_NAME = "latterworks/parsed_ollama_data"
MAX_MATCHES = 15
TIMEOUT_SECONDS = 8  # More stable timeout
RETRY_LIMIT = 3  # Retry on errors

# Define dataset structure
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

    for attempt in range(RETRY_LIMIT):
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS) as response:
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
            print(f"[{datetime.now().isoformat()}] Attempt {attempt+1}/{RETRY_LIMIT} failed for {ip}: {e}")
            await asyncio.sleep(2)  # Short delay before retrying
    return [{
        "IP": ip, "Model": "N/A", "Product": product, "Status": "Failed after retries",
        "ISP": isp, "ASN": asn, "Org": org, "Country": country, "City": city,
        "Longitude": longitude, "Latitude": latitude, "Port": port, "Timestamp": timestamp,
        "Tags": tags, "Transport": transport
    }]

# Define and run main function
async def main():
    api = shodan.Shodan(SHODAN_API_KEY)
    matches = fetch_shodan_matches(api, MAX_MATCHES)
    print(f"[{datetime.now().isoformat()}] Fetched {len(matches)} matches from Shodan")
    
    async with aiohttp.ClientSession() as session:
        tasks = [query_ollama(match, session) for match in matches]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"[{datetime.now().isoformat()}] Async task was cancelled! Retrying...")
            return await main()

        new_data = [entry for sublist in results for entry in sublist if isinstance(sublist, list)]

    with open("temp_data.json", "w") as f:
        json.dump(new_data, f)
    print(f"[{datetime.now().isoformat()}] Saved {len(new_data)} entries to temp_data.json")

    try:
        existing_dataset = load_dataset(REPO_NAME)
        existing_data = existing_dataset["train"].to_list() if "train" in existing_dataset else []
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Dataset not found: {e}")
        existing_data = []

    dataset = Dataset.from_list(existing_data + new_data, features=features)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(REPO_NAME)
    print(f"[{datetime.now().isoformat()}] Dataset updated with {len(new_data)} new entries at: https://huggingface.co/datasets/{REPO_NAME}")

# **Ensure correct execution in Jupyter/Colab**
try:
    asyncio.run(main())  # Works in normal Python environments
except RuntimeError:
    # Fix for Jupyter/Colab's nested event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
