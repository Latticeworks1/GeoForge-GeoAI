# model_matcher.py

import asyncio
import requests
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import httpx
import json
from typing import List, Dict, Union
import nest_asyncio
import argparse # Added argparse
# asyncio is imported below, removing the redundant one here.

# Global cache variables, initialized to None
_IP_MODELS = None
_IP_MODEL_EMBEDDINGS = None
_FUZZY_MODEL = None

# Define query here, can be overridden in main() or passed as argument
query = "Summarize what SmolaAgents is in 2 sentences." # This can remain a global default for the prompt


def load_and_cache_index(dataset_name: str, split: str, sentence_transformer_model_name: str):
    """
    Load the dataset once and build a cached index mapping each IP to its list of model names and
    precomputed embeddings for fuzzy matching.
    
    Returns:
        ip_models: Dict mapping IP (str) -> list of model names (list of str).
        ip_model_embeddings: Dict mapping IP (str) -> Dict of {model_name: embedding (np.array)}.
        fuzzy_model: The SentenceTransformer instance used for generating embeddings.
    """
    global _IP_MODELS, _IP_MODEL_EMBEDDINGS, _FUZZY_MODEL
    if _IP_MODELS is None or _IP_MODEL_EMBEDDINGS is None or _FUZZY_MODEL is None:
        ds = load_dataset(dataset_name)
        train_dataset = ds[split]
        ips = train_dataset["IP"]
        models_lists = train_dataset["Models"]

        # Load the transformer once for fuzzy matching, using the provided model name.
        print(f"Loading SentenceTransformer model: {sentence_transformer_model_name}")
        _FUZZY_MODEL = SentenceTransformer(sentence_transformer_model_name)
        _IP_MODELS = {ip: model_list for ip, model_list in zip(ips, models_lists)}
        _IP_MODEL_EMBEDDINGS = {}
        for ip, model_list in _IP_MODELS.items():
            embeddings = _FUZZY_MODEL.encode(model_list, convert_to_numpy=True)
            _IP_MODEL_EMBEDDINGS[ip] = {name: emb for name, emb in zip(model_list, embeddings)}
    return _IP_MODELS, _IP_MODEL_EMBEDDINGS, _FUZZY_MODEL


def search_best_model(query: str, ip_model_embeddings: dict, fuzzy_model: SentenceTransformer, threshold: float = 0.7):
    """
    Search across all instances to find the best matching model using cosine similarity.
    
    Args:
        query: The fuzzy model name query.
        ip_model_embeddings: Mapping of IP to {model_name: embedding}.
        fuzzy_model: The SentenceTransformer instance.
        threshold: Minimum cosine similarity for a valid match.
    
    Returns:
        Tuple (matched_ip, matched_model, similarity) if a match is found; otherwise, None.
    """
    query_emb = fuzzy_model.encode([query], convert_to_numpy=True)
    best_match = None
    best_similarity = -1.0
    for ip, models_emb in ip_model_embeddings.items():
        for model_name, emb in models_emb.items():
            sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
            if sim > best_similarity and sim >= threshold:
                best_similarity = sim
                best_match = (ip, model_name)
    if best_match:
        return best_match[0], best_match[1], best_similarity
    return None


def call_ollama_chat(
    ip: str,
    model: str,
    messages: List[Dict[str, str]],
    port: int = 11434,
    stream: bool = False,
    timeout: int = 30
) -> Union[str, List[str]]:
    """
    Synchronously call the Ollama chat API and return the assistant's reply.
    
    Args:
        ip (str): The target instance's IP address.
        model (str): The name of the model to use.
        messages (List[Dict[str, str]]): A list of chat messages (e.g., [{"role": "user", "content": "Hello"}]).
        port (int): The port number for the API (default: 11434).
        stream (bool): Whether to enable streaming responses (default: False).
        timeout (int): Timeout for the API request in seconds (default: 30).
    
    Returns:
        Union[str, List[str]]: The assistant's reply as a string for non-streaming calls,
        or a list of streamed responses if streaming is enabled.
    
    Raises:
        RuntimeError: If the API call fails.
    """
    host = f"http://{ip}:{port}"
    url = f"{host}/api/chat"
    payload = {"model": model, "messages": messages, "stream": stream}
    try:
        with requests.post(url, json=payload, timeout=timeout, stream=stream) as response:
            response.raise_for_status()
            if stream:
                replies = []
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = line.decode("utf-8")
                            data = json.loads(chunk)
                            # Try to extract content from either 'message' or top-level 'response'
                            if "message" in data:
                                content = data["message"].get("content", "")
                            elif "response" in data:
                                content = data.get("response", "")
                            else:
                                content = ""
                            if content:
                                replies.append(content)
                                print(content, end="", flush=True)
                        except Exception as e:
                            raise RuntimeError(f"Error parsing streaming chunk: {e}")
                return replies
            # Non-streaming: extract from 'message' or 'response'
            result = response.json()
            if "message" in result:
                return result["message"].get("content", "No response received")
            elif "response" in result:
                return result.get("response", "No response received")
            else:
                return "No response received"
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request to {url} timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP error occurred while calling Ollama chat API at {url}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while calling Ollama chat API at {url}: {e}")


async def stream_response(model: str, query: str, host: str):
    """
    Asynchronously stream chat responses from an Ollama instance.
    If streaming fails, the exception is raised to allow for fallback.
    
    Args:
        model (str): The model name to query.
        query (str): The query text.
        host (str): The full host URL (e.g., "http://<ip>:11434").
    
    Streams:
        Prints response chunks as they arrive.
    """
    try:
        async with httpx.AsyncClient(base_url=host, timeout=30.0) as client:
            print(f"Streaming query to model '{model}': {query}")
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            # Try to extract content from either 'message' or 'response'
                            if "message" in data:
                                content = data["message"].get("content", "")
                            elif "response" in data:
                                content = data.get("response", "")
                            else:
                                content = ""
                            if content:
                                print(content, end="", flush=True)
                        except Exception as e:
                            raise RuntimeError(f"Error parsing streaming line: {e}")
            print("\n[STREAMING COMPLETE]")
    except Exception as e:
        print(f"Error during async streaming: {e}")
        raise


async def main_async_stream(model: str, query: str, host: str):
    """
    Attempt to stream a response asynchronously.
    If streaming fails, fall back to a synchronous call.
    """
    try:
        await stream_response(model, query, host)
    except Exception:
        print("Falling back to synchronous call...")
        loop = asyncio.get_event_loop()
        # Extract IP from host: host is like "http://<ip>:11434"
        ip = host.split("://")[1].split(":")[0]
        sync_response = await loop.run_in_executor(
            None, call_ollama_chat, ip, model, [{"role": "user", "content": query}]
        )
        print("Synchronous response:", sync_response)


def main(args):
    # Load the dataset and build the index once (cached globally)
    global _IP_MODELS, _IP_MODEL_EMBEDDINGS, _FUZZY_MODEL # Ensure we are using the global cache
    
    # Load and cache index - this call populates the global variables
    # Pass dataset_name, split, and sentence_transformer_model from args
    ip_models, ip_model_embeddings, fuzzy_model = load_and_cache_index(
        dataset_name=args.dataset_name,
        split=args.split,
        sentence_transformer_model_name=args.sentence_transformer_model
    )

    # Perform a fuzzy search for the desired model using args.initial_search_model
    result = search_best_model(args.initial_search_model, ip_model_embeddings, fuzzy_model, threshold=0.7) # Threshold could also be an arg
    
    if result is None:
        print(f"No matching model found for '{args.initial_search_model}' above the threshold.")
        print("Cannot proceed without a matched model and host.")
        return

    matched_ip, matched_model, sim = result
    print(f"Matched IP: {matched_ip}, Model: {matched_model} for query '{args.initial_search_model}' (Similarity: {sim:.3f})")
    host = f"http://{matched_ip}:11434" # Port could also be an arg

    # Use the global 'query' for the actual summarization task, or make it an arg too
    current_query_for_ollama = query 
    if args.ollama_query: # Allow overriding the global query via CLI
        current_query_for_ollama = args.ollama_query
    
    print(f"Using Ollama query: \"{current_query_for_ollama}\"")

    # First, try a synchronous call
    try:
        sync_reply = call_ollama_chat(matched_ip, matched_model, [{"role": "user", "content": current_query_for_ollama}])
        print("Synchronous Chat response:", sync_reply)
    except RuntimeError as e:
        print(f"Error during synchronous call: {e}")

    # Then, attempt asynchronous streaming with auto-fallback
    # Ensure nest_asyncio is applied if running in a notebook-like environment
    if 'nest_asyncio' in globals():
        nest_asyncio.apply()
    asyncio.run(main_async_stream(matched_model, current_query_for_ollama, host))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Matcher: Find and interact with Ollama models.")
    parser.add_argument("--dataset_name", type=str, default="latterworks/instances", help="Name of the Hugging Face dataset to load for IP and model indexing.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test').")
    parser.add_argument("--initial_search_model", type=str, default="deepseek", help="Name of the model to search for initially to find a host.")
    parser.add_argument("--sentence_transformer_model", type=str, default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use for fuzzy matching.")
    parser.add_argument("--ollama_query", type=str, default=query, help="The query to send to the matched Ollama model.") # Uses global query as default
    
    args = parser.parse_args()
    main(args)
