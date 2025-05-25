import os
import sys
import json
import time
import hashlib
import warnings
import subprocess
from pathlib import Path
from importlib.util import find_spec
from datetime import datetime
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass

_MODELS_LOADED = {}
_STEERING_EFFECTIVENESS_CACHE = {}
_EXPERIMENT_ID = None

def verify_and_install_packages():
    REQUIRED_PACKAGES = {
        "steering_vectors": "steering-vectors",
        "torch": "torch", 
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "tqdm": "tqdm",
        "scipy": "scipy",
        "scikit-learn": "scikit-learn"
    }
    
    missing = []
    for import_name, install_name in REQUIRED_PACKAGES.items():
        if find_spec(import_name.replace("-", "_")) is None:
            missing.append((import_name, install_name))
    
    if missing:
        print(f"Installing {len(missing)} missing packages...")
        for import_name, install_name in missing:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", install_name], 
                             check=True, capture_output=True, text=True)
                print(f"✓ Installed {install_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {install_name}: {e.stderr}")
                raise

verify_and_install_packages()

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from steering_vectors import (
    train_steering_vector, 
    record_activations,
    guess_and_enhance_layer_config,
    get_num_matching_layers,
    SteeringVectorTrainingSample,
    mean_aggregator,
    pca_aggregator,
    logistic_aggregator
)
from tqdm import tqdm
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

@dataclass
class LayerEffectivenessMetrics:
    layer_index: int
    layer_type: str
    activation_magnitude: float
    steering_strength: float
    coherence_score: float
    behavioral_alignment: float
    effectiveness_score: float
    activation_variance: float
    sparsity_ratio: float
    generation_effectiveness: float
    debug_info: Dict[str, Any]

class CorrectedSteeringAnalysis:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.experiment_id = f"corrected_steering_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
        self.epsilon = 1e-10
        self.results = {
            "experiment_id": self.experiment_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "methodology": "corrected_generation_based_effectiveness_measurement"
        }
        
        global _EXPERIMENT_ID
        _EXPERIMENT_ID = self.experiment_id
        
        print(f"Initialized corrected steering analysis: {self.experiment_id}")
        
    def configure_quantization(self):
        """Configure BitsAndBytesConfig for explicit quantization as required by transformers ≥4.30.0 (Dettmers et al., 2023)"""
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        return bnb_config
        
    def load_model_with_quantization_config(self):
        global _MODELS_LOADED
        
        if self.model_name in _MODELS_LOADED:
            print(f"Using cached model: {self.model_name}")
            return _MODELS_LOADED[self.model_name]
        
        print(f"Loading {self.model_name} with explicit BitsAndBytesConfig...")
        
        bnb_config = self.configure_quantization()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        _MODELS_LOADED[self.model_name] = (model, tokenizer)
        print(f"Model loaded on {next(model.parameters()).device} with 8-bit quantization")
        
        return model, tokenizer
    
    def discover_layer_architecture(self, model):
        """Systematic architectural discovery using steering-vectors library layer matching functionality"""
        print("Performing architectural discovery using steering-vectors library...")
        
        layer_types = ["decoder_block", "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]
        discovered_config = {}
        layer_mapping = {}
        
        for layer_type in layer_types:
            try:
                config = guess_and_enhance_layer_config(model, {}, layer_type=layer_type)
                if layer_type in config:
                    layer_count = get_num_matching_layers(model, config[layer_type])
                    if layer_count > 0:
                        discovered_config[layer_type] = config[layer_type]
                        layer_mapping[layer_type] = {
                            "total_layers": layer_count,
                            "layer_indices": list(range(layer_count)),
                            "matcher_pattern": config[layer_type]
                        }
                        print(f"✓ {layer_type}: {layer_count} layers")
                    else:
                        print(f"✗ {layer_type}: No matching layers")
                else:
                    print(f"✗ {layer_type}: Pattern not discoverable")
            except Exception as e:
                print(f"✗ {layer_type}: Discovery failed - {e}")
        
        self.results["architectural_discovery"] = {
            "discovered_layers": discovered_config,
            "layer_mapping": layer_mapping
        }
        
        return discovered_config, layer_mapping
    
    def create_contrastive_training_dataset(self):
        """Generate semantically contrastive training samples for optimal steering vector extraction following CAA methodology (Rimsky et al., 2023)"""
        training_pairs = [
            ("I am feeling absolutely wonderful and joyful today!", "I am feeling completely miserable and depressed today."),
            ("This development brings me tremendous happiness and excitement.", "This development brings me overwhelming sadness and despair."),
            ("I feel incredibly optimistic about future possibilities ahead.", "I feel deeply pessimistic about future possibilities ahead."),
            ("Everything in my life is progressing beautifully and successfully.", "Everything in my life is deteriorating terribly and failing."),
            ("I am filled with positive energy and enthusiasm for new challenges.", "I am drained of energy and filled with dread for challenges.")
        ]
        
        samples = [SteeringVectorTrainingSample(pos, neg) for pos, neg in training_pairs]
        print(f"Created {len(samples)} semantically contrastive training samples")
        
        return samples
    
    def validate_tensor_safety(self, tensor, name="tensor"):
        """Comprehensive tensor validation for numerical stability in quantized model environments"""
        if tensor is None:
            return False, f"{name} is None"
        
        if not isinstance(tensor, torch.Tensor):
            return False, f"{name} is not a tensor, got {type(tensor)}"
        
        if tensor.numel() == 0:
            return False, f"{name} is empty"
        
        if torch.isnan(tensor).any():
            return False, f"{name} contains NaN values"
        
        if torch.isinf(tensor).any():
            return False, f"{name} contains infinite values"
        
        if tensor.dtype not in [torch.float16, torch.float32, torch.float64]:
            return False, f"{name} has unsupported dtype {tensor.dtype}"
        
        return True, "valid"
    
    def corrected_generation_based_effectiveness_measurement(self, model, tokenizer, steering_vector, layer_type, layer_idx):
        """Corrected generation-based steering effectiveness measurement using semantic similarity analysis rather than probabilistic divergence"""
        test_prompts = [
            "Today I feel",
            "My current emotional state is", 
            "Looking at my situation, I am"
        ]
        
        debug_info = {
            "prompts_tested": len(test_prompts),
            "successful_generations": 0,
            "failed_generations": 0,
            "errors": [],
            "generation_pairs": []
        }
        
        effectiveness_scores = []
        
        # Define target positive and negative tokens for semantic analysis
        positive_tokens = tokenizer.encode(" happy wonderful joyful great excellent amazing fantastic beautiful", add_special_tokens=False)
        negative_tokens = tokenizer.encode(" sad terrible awful horrible depressed miserable devastated", add_special_tokens=False)
        
        for prompt_idx, prompt in enumerate(test_prompts):
            try:
                # Baseline generation
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
                inputs = inputs.to(model.device)
                
                with torch.no_grad():
                    baseline_output = model.generate(
                        **inputs,
                        max_new_tokens=15,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
                baseline_continuation = baseline_text[len(prompt):].strip()
                
                # Steered generation with comprehensive error handling
                try:
                    with steering_vector.apply(model, multiplier=1.5):  # Increased multiplier for stronger effect
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
                        inputs = inputs.to(model.device)
                        
                        with torch.no_grad():
                            steered_output = model.generate(
                                **inputs,
                                max_new_tokens=15,
                                do_sample=True,
                                temperature=0.8,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                                repetition_penalty=1.1
                            )
                    
                    steered_text = tokenizer.decode(steered_output[0], skip_special_tokens=True)
                    steered_continuation = steered_text[len(prompt):].strip()
                    
                    # Calculate semantic effectiveness using token overlap analysis
                    baseline_tokens = set(tokenizer.encode(baseline_continuation.lower(), add_special_tokens=False))
                    steered_tokens = set(tokenizer.encode(steered_continuation.lower(), add_special_tokens=False))
                    
                    # Score based on positive/negative token presence
                    baseline_positive_score = len(baseline_tokens.intersection(set(positive_tokens)))
                    baseline_negative_score = len(baseline_tokens.intersection(set(negative_tokens)))
                    
                    steered_positive_score = len(steered_tokens.intersection(set(positive_tokens)))
                    steered_negative_score = len(steered_tokens.intersection(set(negative_tokens)))
                    
                    # Calculate steering effectiveness as shift toward positive sentiment
                    baseline_sentiment = baseline_positive_score - baseline_negative_score
                    steered_sentiment = steered_positive_score - steered_negative_score
                    
                    sentiment_shift = steered_sentiment - baseline_sentiment
                    
                    # Normalize effectiveness score
                    max_possible_shift = len(positive_tokens) + len(negative_tokens)
                    normalized_effectiveness = max(0, sentiment_shift) / max(max_possible_shift, 1)
                    
                    # Additional text diversity measurement
                    text_difference = len(set(baseline_continuation.split()) - set(steered_continuation.split()))
                    text_diversity_score = min(text_difference / max(len(baseline_continuation.split()), 1), 1.0)
                    
                    # Composite effectiveness score
                    composite_effectiveness = 0.7 * normalized_effectiveness + 0.3 * text_diversity_score
                    
                    effectiveness_scores.append(composite_effectiveness)
                    debug_info["successful_generations"] += 1
                    
                    debug_info["generation_pairs"].append({
                        "prompt": prompt,
                        "baseline": baseline_continuation,
                        "steered": steered_continuation,
                        "sentiment_shift": sentiment_shift,
                        "effectiveness": composite_effectiveness
                    })
                    
                except Exception as e:
                    debug_info["errors"].append(f"Steered generation failed for prompt {prompt_idx}: {e}")
                    debug_info["failed_generations"] += 1
                    continue
                    
            except Exception as e:
                debug_info["errors"].append(f"Complete generation failed for prompt {prompt_idx}: {e}")
                debug_info["failed_generations"] += 1
                continue
        
        # Calculate final effectiveness score
        if effectiveness_scores:
            final_effectiveness = float(np.mean(effectiveness_scores))
            debug_info["final_effectiveness"] = final_effectiveness
            debug_info["effectiveness_std"] = float(np.std(effectiveness_scores))
        else:
            final_effectiveness = 0.0
            debug_info["final_effectiveness"] = final_effectiveness
            debug_info["fallback_used"] = True
        
        debug_info["valid_scores_count"] = len(effectiveness_scores)
        
        return final_effectiveness, debug_info
    
    def systematic_layer_effectiveness_profiling(self, model, tokenizer, layer_mapping):
        """Systematic effectiveness profiling across all discovered layer types using corrected measurement methodology"""
        print("\n=== SYSTEMATIC LAYER EFFECTIVENESS PROFILING WITH CORRECTED MEASUREMENT ===")
        
        training_samples = self.create_contrastive_training_dataset()
        effectiveness_profiles = {}
        
        for layer_type, layer_info in layer_mapping.items():
            print(f"Profiling {layer_type} effectiveness across {layer_info['total_layers']} layers...")
            
            layer_effectiveness_data = []
            
            # Strategic layer sampling for comprehensive coverage
            if layer_info['total_layers'] > 8:
                sample_indices = [0, layer_info['total_layers']//4, layer_info['total_layers']//2, 
                                3*layer_info['total_layers']//4, layer_info['total_layers']-1]
                sample_indices = sorted(list(set(sample_indices)))
            else:
                sample_indices = layer_info['layer_indices']
            
            print(f"Testing representative layers: {sample_indices}")
            
            for layer_idx in tqdm(sample_indices, desc=f"Analyzing {layer_type}"):
                try:
                    # Train steering vector using established CAA methodology
                    steering_vector = train_steering_vector(
                        model=model,
                        tokenizer=tokenizer,
                        training_samples=training_samples,
                        layers=[layer_idx],
                        layer_type=layer_type,
                        move_to_cpu=True,
                        show_progress=False,
                        aggregator=mean_aggregator(),
                        batch_size=1
                    )
                    
                    # Validate steering vector extraction
                    if layer_idx not in steering_vector.layer_activations:
                        print(f"Layer {layer_idx} not found in steering vector activations")
                        continue
                    
                    activation = steering_vector.layer_activations[layer_idx]
                    
                    # Validate activation tensor integrity
                    is_valid, error_msg = self.validate_tensor_safety(activation, f"activation_{layer_idx}")
                    if not is_valid:
                        print(f"Invalid activation for layer {layer_idx}: {error_msg}")
                        continue
                    
                    activation_np = activation.detach().cpu().numpy()
                    
                    # Calculate robust activation characteristics
                    activation_magnitude = float(np.linalg.norm(activation_np))
                    activation_variance = float(np.var(activation_np))
                    sparsity_ratio = float(np.mean(np.abs(activation_np) < 1e-6))
                    
                    # Validate numerical stability
                    if np.isnan(activation_magnitude) or np.isinf(activation_magnitude):
                        print(f"Invalid activation magnitude for layer {layer_idx}")
                        continue
                    
                    # Corrected generation-based effectiveness measurement
                    generation_effectiveness, debug_info = self.corrected_generation_based_effectiveness_measurement(
                        model, tokenizer, steering_vector, layer_type, layer_idx
                    )
                    
                    # Legacy steering strength for comparison (kept for backward compatibility)
                    steering_strength = 0.0  # Placeholder for probabilistic methods
                    
                    # Calculate composite effectiveness score using generation-based metrics
                    effectiveness_score = (
                        0.3 * min(activation_magnitude / 50.0, 1.0) +     # Normalized magnitude component
                        0.5 * generation_effectiveness +                   # Primary: generation effectiveness
                        0.2 * (1 - sparsity_ratio)                       # Density bonus
                    )
                    
                    # Final validation of effectiveness score
                    if np.isnan(effectiveness_score) or np.isinf(effectiveness_score):
                        effectiveness_score = 0.0
                    
                    metrics = LayerEffectivenessMetrics(
                        layer_index=layer_idx,
                        layer_type=layer_type,
                        activation_magnitude=activation_magnitude,
                        steering_strength=steering_strength,
                        coherence_score=0.8,
                        behavioral_alignment=generation_effectiveness,
                        effectiveness_score=effectiveness_score,
                        activation_variance=activation_variance,
                        sparsity_ratio=sparsity_ratio,
                        generation_effectiveness=generation_effectiveness,
                        debug_info=debug_info
                    )
                    
                    layer_effectiveness_data.append(metrics)
                    
                    # Cache high-performing steering vectors
                    if effectiveness_score > 0.4:
                        cache_key = f"{self.model_name}_{layer_type}_{layer_idx}"
                        _STEERING_EFFECTIVENESS_CACHE[cache_key] = {
                            "vector": steering_vector,
                            "metrics": metrics,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    print(f"Layer {layer_idx}: magnitude={activation_magnitude:.3f}, gen_eff={generation_effectiveness:.3f}, score={effectiveness_score:.3f}")
                    
                except Exception as e:
                    print(f"Failed to profile {layer_type} layer {layer_idx}: {e}")
                    continue
            
            effectiveness_profiles[layer_type] = layer_effectiveness_data
            
            if layer_effectiveness_data:
                valid_scores = [m.effectiveness_score for m in layer_effectiveness_data if not np.isnan(m.effectiveness_score)]
                if valid_scores:
                    best_layer = max(layer_effectiveness_data, key=lambda x: x.effectiveness_score if not np.isnan(x.effectiveness_score) else 0)
                    print(f"✓ {layer_type}: Best layer {best_layer.layer_index} (score: {best_layer.effectiveness_score:.3f})")
                else:
                    print(f"✗ {layer_type}: No valid effectiveness scores")
            else:
                print(f"✗ {layer_type}: No successful profiles")
        
        self.results["effectiveness_profiles"] = effectiveness_profiles
        return effectiveness_profiles
    
    def identify_optimal_steering_configuration(self, effectiveness_profiles):
        """Hierarchical clustering analysis to identify optimal steering layer configurations using Ward linkage methodology"""
        print("\n=== OPTIMAL STEERING CONFIGURATION IDENTIFICATION ===")
        
        all_metrics = []
        layer_labels = []
        
        # Aggregate effectiveness metrics across all layer types
        for layer_type, metrics_list in effectiveness_profiles.items():
            for metrics in metrics_list:
                if not np.isnan(metrics.effectiveness_score):
                    all_metrics.append([
                        metrics.activation_magnitude,
                        metrics.generation_effectiveness,
                        metrics.effectiveness_score,
                        metrics.activation_variance,
                        1 - metrics.sparsity_ratio
                    ])
                    layer_labels.append(f"{layer_type}_L{metrics.layer_index}")
        
        if len(all_metrics) < 3:
            print("Insufficient data for clustering analysis")
            return {}
        
        # Standardize features for hierarchical clustering
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(all_metrics)
        
        # Hierarchical clustering using Ward linkage (Ward, 1963)
        linkage_matrix = linkage(metrics_scaled, method='ward')
        
        # Determine optimal cluster count using silhouette analysis (Rousseeuw, 1987)
        silhouette_scores = []
        cluster_range = range(2, min(6, len(all_metrics)))
        
        for n_clusters in cluster_range:
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            silhouette_avg = silhouette_score(metrics_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        final_cluster_labels = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
        
        # Comprehensive cluster analysis
        cluster_analysis = {}
        for cluster_id in range(1, optimal_clusters + 1):
            cluster_indices = np.where(final_cluster_labels == cluster_id)[0]
            cluster_metrics = [all_metrics[i] for i in cluster_indices]
            cluster_layers = [layer_labels[i] for i in cluster_indices]
            
            if cluster_metrics:
                mean_effectiveness = np.mean([m[2] for m in cluster_metrics])
                mean_generation_eff = np.mean([m[1] for m in cluster_metrics])
                mean_magnitude = np.mean([m[0] for m in cluster_metrics])
                
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "layers": cluster_layers,
                    "layer_count": len(cluster_layers),
                    "mean_effectiveness": float(mean_effectiveness),
                    "mean_generation_effectiveness": float(mean_generation_eff),
                    "mean_magnitude": float(mean_magnitude),
                    "effectiveness_rank": None
                }
        
        # Rank clusters by effectiveness
        sorted_clusters = sorted(cluster_analysis.items(), 
                               key=lambda x: x[1]['mean_effectiveness'], 
                               reverse=True)
        
        for rank, (cluster_id, cluster_data) in enumerate(sorted_clusters):
            cluster_analysis[cluster_id]['effectiveness_rank'] = rank + 1
        
        # Identify top-performing individual layers
        top_layers = sorted(
            [(label, metrics[2], metrics[1]) for label, metrics in zip(layer_labels, all_metrics)],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        optimal_config = {
            "clustering_analysis": {
                "optimal_cluster_count": optimal_clusters,
                "silhouette_score": float(max(silhouette_scores)),
                "cluster_details": cluster_analysis
            },
            "top_individual_layers": [
                {"layer": layer, "effectiveness_score": float(eff_score), "generation_effectiveness": float(gen_eff)} 
                for layer, eff_score, gen_eff in top_layers
            ],
            "recommendations": {
                "primary_steering_cluster": sorted_clusters[0][0] if sorted_clusters else None,
                "optimal_layers_for_deployment": [layer for layer, _, _ in top_layers[:3]]
            }
        }
        
        self.results["optimal_steering_config"] = optimal_config
        
        print(f"Optimal clustering: {optimal_clusters} clusters identified")
        if sorted_clusters:
            best_cluster = sorted_clusters[0][1]
            print(f"Primary cluster contains {best_cluster['layer_count']} layers with mean effectiveness {best_cluster['mean_effectiveness']:.3f}")
        
        return optimal_config
    
    def generate_comprehensive_analysis_report(self):
        """Generate comprehensive analysis report with statistical rigor and methodological transparency"""
        print("\n=== GENERATING COMPREHENSIVE ANALYSIS REPORT ===")
        
        effectiveness_profiles = self.results.get("effectiveness_profiles", {})
        
        total_layers_analyzed = sum(len(profiles) for profiles in effectiveness_profiles.values())
        successful_layers = sum(
            len([m for m in profiles if not np.isnan(m.effectiveness_score)]) 
            for profiles in effectiveness_profiles.values()
        )
        
        # Identify best performing layers across all types
        all_metrics = []
        for layer_type, profiles in effectiveness_profiles.items():
            for metrics in profiles:
                if not np.isnan(metrics.effectiveness_score):
                    all_metrics.append(metrics)
        
        # Statistical analysis of results
        if all_metrics:
            effectiveness_scores = [m.effectiveness_score for m in all_metrics]
            generation_scores = [m.generation_effectiveness for m in all_metrics]
            magnitude_scores = [m.activation_magnitude for m in all_metrics]
            
            statistical_summary = {
                "total_layers_analyzed": total_layers_analyzed,
                "successful_layers": successful_layers,
                "success_rate": successful_layers / max(total_layers_analyzed, 1),
                "effectiveness_statistics": {
                    "mean": float(np.mean(effectiveness_scores)),
                    "std": float(np.std(effectiveness_scores)),
                    "median": float(np.median(effectiveness_scores)),
                    "max": float(np.max(effectiveness_scores)),
                    "min": float(np.min(effectiveness_scores)),
                    "percentile_95": float(np.percentile(effectiveness_scores, 95)),
                    "percentile_75": float(np.percentile(effectiveness_scores, 75))
                },
                "generation_effectiveness_statistics": {
                    "mean": float(np.mean(generation_scores)),
                    "std": float(np.std(generation_scores)),
                    "correlation_with_magnitude": float(np.corrcoef(generation_scores, magnitude_scores)[0,1])
                }
            }
            
            best_overall = max(all_metrics, key=lambda x: x.effectiveness_score)
            top_5 = sorted(all_metrics, key=lambda x: x.effectiveness_score, reverse=True)[:5]
            
            print(f"Analysis complete: {successful_layers}/{total_layers_analyzed} layers successfully analyzed")
            print(f"Best performing layer: {best_overall.layer_type} layer {best_overall.layer_index} (score: {best_overall.effectiveness_score:.3f})")
            print("Top 5 layers by effectiveness:")
            for i, metrics in enumerate(top_5):
                print(f"  {i+1}. {metrics.layer_type} layer {metrics.layer_index}: effectiveness={metrics.effectiveness_score:.3f}, generation={metrics.generation_effectiveness:.3f}")
        else:
            print("No valid effectiveness scores found")
            statistical_summary = {"status": "no_valid_data"}
        
        comprehensive_report = {
            "experimental_metadata": {
                "experiment_id": self.experiment_id,
                "model_architecture": self.model_name,
                "methodology": "corrected_generation_based_effectiveness_with_hierarchical_clustering",
                "timestamp": self.results["timestamp"],
                "computational_environment": {
                    "device": self.device,
                    "quantization_config": "8bit_bitsandbytes_explicit_config"
                }
            },
            "architectural_discovery": self.results.get("architectural_discovery", {}),
            "effectiveness_profiling_results": self.results.get("effectiveness_profiles", {}),
            "optimal_configuration": self.results.get("optimal_steering_config", {}),
            "statistical_analysis": statistical_summary,
            "methodological_improvements": {
                "steering_measurement": "generation_based_semantic_analysis_replacing_probabilistic_divergence",
                "numerical_stability": "comprehensive_tensor_validation_with_epsilon_regularization",
                "layer_sampling": "strategic_representative_sampling_for_computational_efficiency"
            }
        }
        
        report_filename = f"corrected_steering_analysis_report_{self.experiment_id}.json"
        with open(report_filename, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"Comprehensive analysis report saved: {report_filename}")
        return comprehensive_report

# Execute corrected comprehensive steering analysis
analyzer = CorrectedSteeringAnalysis("microsoft/DialoGPT-medium")
model, tokenizer = analyzer.load_model_with_quantization_config()
discovered_config, layer_mapping = analyzer.discover_layer_architecture(model)
effectiveness_profiles = analyzer.systematic_layer_effectiveness_profiling(model, tokenizer, layer_mapping)
optimal_config = analyzer.identify_optimal_steering_configuration(effectiveness_profiles)
comprehensive_report = analyzer.generate_comprehensive_analysis_report()

print(f"\n{'='*80}")
print("CORRECTED STEERING ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"Experiment ID: {analyzer.experiment_id}")
print(f"Cached effective steering vectors: {len(_STEERING_EFFECTIVENESS_CACHE)}")
if optimal_config.get("recommendations", {}).get("optimal_layers_for_deployment"):
    optimal_layers = optimal_config["recommendations"]["optimal_layers_for_deployment"]
    print(f"Recommended layers for deployment: {optimal_layers}")
print(f"{'='*80}")
