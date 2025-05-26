"""
Vectsteer: Advanced Steering Vectors API
========================================

A powerful, intuitive wrapper around the steering_vectors library that provides:
- Fluent API for easy steering configuration
- Conceptor-based steering with boolean operations
- Support for multiple steering approaches (CAA, Style Vectors, LaRS)
- Comprehensive layer targeting options
- Save/load functionality for trained vectors

Built following Hugging Face design principles:
- Prioritize accessibility and simple APIs
- Abstract complexity behind intuitive interfaces
- Provide sensible defaults with progressive disclosure
- Enable easy sharing and extensibility
"""

import torch
from typing import List, Tuple, Union, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Import steering_vectors components
from steering_vectors import (
    Aggregator, 
    mean_aggregator, 
    pca_aggregator,
    logistic_aggregator,
    LayerType,
    LayerMatcher,
    ModelLayerConfig,
    get_num_matching_layers,
    guess_and_enhance_layer_config,
    record_activations,
    ablation_operator,
    ablation_then_addition_operator,
    addition_operator,
    PatchDeltaOperator,
    SteeringPatchHandle,
    SteeringVector,
    SteeringVectorTrainingSample,
    aggregate_activations,
    extract_activations,
    train_steering_vector
)


class AggregatorType(str, Enum):
    """Available aggregation methods for training steering vectors."""
    MEAN = "mean"
    PCA = "pca" 
    LOGISTIC = "logistic"


class OperatorType(str, Enum):
    """Available operators for applying steering vectors."""
    ADDITION = "addition"
    ABLATION = "ablation"
    ABLATION_THEN_ADDITION = "ablation_then_addition"


@dataclass
class SteeringDirection:
    """Represents a steering direction with examples and parameters."""
    name: str
    examples: List[Tuple[str, str]]
    strength: float = 1.0
    read_token_index: int = -1
    
    # Will be filled during training
    vector: Optional[SteeringVector] = None
    
    def invert(self) -> 'SteeringDirection':
        """Create an inverted version of this direction."""
        inverted_examples = [(neg, pos) for pos, neg in self.examples]
        return SteeringDirection(
            name=f"not_{self.name}",
            examples=inverted_examples,
            strength=-self.strength,
            read_token_index=self.read_token_index
        )


class Vectsteer:
    """
    Fluent API for steering language models using activation engineering.
    
    This class provides an intuitive interface for applying various steering
    techniques including Contrastive Activation Addition (CAA), Style Vectors,
    and Conceptor-based steering.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from vectsteer import Vectsteer
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> 
        >>> steerer = Vectsteer(model, tokenizer)
        >>> result = (steerer
        ...     .want("positive", examples=[("I love this", "I hate this")])
        ...     .avoid("technical", examples=[("simple", "technical jargon")])
        ...     .using(layer_type="mlp", layer_nums=[12, 13, 14])
        ...     .run("Write a review:")
        ... )
    """
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase,
        layer_config: Optional[ModelLayerConfig] = None
    ):
        """
        Initialize the Vectsteer with a model and tokenizer.
        
        Args:
            model: The language model to steer
            tokenizer: Tokenizer for the model
            layer_config: Optional custom layer configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_config = layer_config or {}
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # State for fluent API
        self._directions: List[SteeringDirection] = []
        self._layer_type: LayerType = "decoder_block"
        self._layer_nums: Optional[List[int]] = None
        self._aggregator_type: AggregatorType = AggregatorType.MEAN
        self._operator_type: OperatorType = OperatorType.ADDITION
        self._batch_size: int = 1
        self._conceptors: Dict[str, Tensor] = {}
        self._alpha: float = 0.1  # Aperture for conceptors
        
        # Infer and enhance layer config
        self._enhanced_layer_config = guess_and_enhance_layer_config(model, self.layer_config)
    
    def want(self, name: str, examples: List[Tuple[str, str]], strength: float = 1.0) -> 'Vectsteer':
        """
        Add a steering direction to favor.
        
        Args:
            name: Name for this steering direction
            examples: List of (positive, negative) example pairs
            strength: How strongly to apply this direction (default: 1.0)
        
        Returns:
            Self for chaining
        """
        direction = SteeringDirection(name=name, examples=examples, strength=strength)
        self._directions.append(direction)
        return self
    
    def avoid(self, name: str, examples: List[Tuple[str, str]], strength: float = 1.0) -> 'Vectsteer':
        """
        Add a steering direction to avoid.
        
        Args:
            name: Name for this steering direction  
            examples: List of (positive, negative) example pairs (will be inverted)
            strength: How strongly to avoid this direction (default: 1.0)
        
        Returns:
            Self for chaining
        """
        direction = SteeringDirection(name=name, examples=examples, strength=-strength)
        self._directions.append(direction)
        return self
    
    def using(
        self, 
        layer_type: LayerType = "decoder_block", 
        layer_nums: Optional[List[int]] = None
    ) -> 'Vectsteer':
        """
        Specify which layers to use for steering.
        
        Layer Type Guide:
        - "decoder_block": Controls final output style and tone
        - "mlp": Adjusts reasoning and cognitive style  
        - "self_attn": Modifies attention patterns and focus
        - "input_layernorm": Affects initial input interpretation
        - "post_attention_layernorm": Controls retained information
        
        Args:
            layer_type: Type of layer to target
            layer_nums: Specific layer numbers (default: None, uses optimal layers)
        
        Returns:
            Self for chaining
        """
        self._layer_type = layer_type
        self._layer_nums = layer_nums
        return self
    
    def with_aggregator(self, aggregator_type: Union[str, AggregatorType]) -> 'Vectsteer':
        """
        Specify how to aggregate activations into steering vectors.
        
        Args:
            aggregator_type: "mean" (default), "pca", or "logistic"
        
        Returns:
            Self for chaining
        """
        if isinstance(aggregator_type, str):
            aggregator_type = AggregatorType(aggregator_type)
        self._aggregator_type = aggregator_type
        return self
    
    def with_operator(self, operator_type: Union[str, OperatorType]) -> 'Vectsteer':
        """
        Specify how to apply steering vectors to model activations.
        
        Args:
            operator_type: "addition" (default), "ablation", or "ablation_then_addition"
        
        Returns:
            Self for chaining
        """
        if isinstance(operator_type, str):
            operator_type = OperatorType(operator_type)
        self._operator_type = operator_type
        return self
    
    def strength(self, global_multiplier: float) -> 'Vectsteer':
        """
        Set a global strength multiplier for all steering directions.
        
        Args:
            global_multiplier: Global multiplier to apply to all steering strengths
        
        Returns:
            Self for chaining
        """
        for direction in self._directions:
            direction.strength *= global_multiplier
        return self
    
    def batch_size(self, size: int) -> 'Vectsteer':
        """
        Set batch size for training (larger = faster if GPU has memory).
        
        Args:
            size: Batch size for training
            
        Returns:
            Self for chaining
        """
        self._batch_size = size
        return self
    
    def _get_aggregator(self) -> Aggregator:
        """Get the appropriate aggregator function."""
        if self._aggregator_type == AggregatorType.MEAN:
            return mean_aggregator()
        elif self._aggregator_type == AggregatorType.PCA:
            return pca_aggregator()
        elif self._aggregator_type == AggregatorType.LOGISTIC:
            return logistic_aggregator()
        else:
            raise ValueError(f"Unknown aggregator type: {self._aggregator_type}")
    
    def _get_operator(self) -> PatchDeltaOperator:
        """Get the appropriate operator function."""
        if self._operator_type == OperatorType.ADDITION:
            return addition_operator()
        elif self._operator_type == OperatorType.ABLATION:
            return ablation_operator()
        elif self._operator_type == OperatorType.ABLATION_THEN_ADDITION:
            return ablation_then_addition_operator()
        else:
            raise ValueError(f"Unknown operator type: {self._operator_type}")
    
    def _auto_select_layers(self) -> List[int]:
        """Automatically select optimal layers based on model size."""
        total_layers = get_num_matching_layers(self.model, self._enhanced_layer_config[self._layer_type])
        
        # Use heuristics based on model size for Llama-style models
        if total_layers <= 12:  # Small models (7B)
            return [total_layers // 2, total_layers // 2 + 1, total_layers // 2 + 2]
        elif total_layers <= 24:  # Medium models (13B)
            return [total_layers // 2 + 2, total_layers // 2 + 3, total_layers // 2 + 4]
        else:  # Large models (30B+)
            return [total_layers // 2 + 4, total_layers // 2 + 5, total_layers // 2 + 6]
    
    def prepare(self, show_progress: bool = True) -> 'Vectsteer':
        """
        Train steering vectors for all configured directions.
        
        Args:
            show_progress: Whether to show training progress
        
        Returns:
            Self for chaining
        """
        # Auto-select layers if not specified
        if self._layer_nums is None:
            self._layer_nums = self._auto_select_layers()
            if show_progress:
                print(f"🎯 Auto-selected layers: {self._layer_nums} for {self._layer_type}")
        
        for direction in self._directions:
            # Convert examples to SteeringVectorTrainingSample objects
            training_samples = [
                SteeringVectorTrainingSample(
                    pos, neg, 
                    read_positive_token_index=direction.read_token_index,
                    read_negative_token_index=direction.read_token_index
                )
                for pos, neg in direction.examples
            ]
            
            # Train the steering vector
            direction.vector = train_steering_vector(
                model=self.model,
                tokenizer=self.tokenizer,
                training_samples=training_samples,
                layers=self._layer_nums,
                layer_type=self._layer_type,
                layer_config=self._enhanced_layer_config,
                aggregator=self._get_aggregator(),
                batch_size=self._batch_size,
                show_progress=show_progress,
                tqdm_desc=f"Training '{direction.name}' vector"
            )
        
        return self
    
    def run(
        self, 
        prompt: str, 
        min_token_index: int = 0,
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        **generate_kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with the configured steering.
        
        Args:
            prompt: Text prompt to generate from
            min_token_index: Minimum token index to apply steering (default: 0)
            max_new_tokens: Maximum new tokens to generate (default: 100)
            num_return_sequences: Number of sequences to generate (default: 1)
            do_sample: Whether to use sampling (default: True)
            temperature: Sampling temperature (default: 0.7)
            **generate_kwargs: Additional kwargs for model.generate()
            
        Returns:
            Generated text(s) with steering applied
        """
        # Ensure vectors are trained
        if any(direction.vector is None for direction in self._directions):
            self.prepare()
        
        # Create active handles list for cleanup
        active_handles = []
        
        try:
            # Apply each steering vector
            for direction in self._directions:
                handle = direction.vector.patch_activations(
                    model=self.model,
                    layer_config=self._enhanced_layer_config,
                    operator=self._get_operator(),
                    multiplier=direction.strength,
                    min_token_index=min_token_index
                )
                active_handles.append(handle)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Set up generation parameters
            generation_config = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.eos_token_id,
                **generate_kwargs
            }
            
            # Generate with steering applied
            with torch.no_grad():
                outputs = self.model.generate(**generation_config)
            
            # Decode outputs and remove the input prompt
            generated_texts = []
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the input prompt from the generated text
                new_text = full_text[len(prompt):].strip()
                generated_texts.append(new_text)
            
            # Return single string if only one sequence requested
            return generated_texts[0] if num_return_sequences == 1 else generated_texts
            
        finally:
            # Clean up by removing all hooks
            for handle in active_handles:
                handle.remove()
    
    def compare(
        self, 
        prompt: str,
        multipliers: List[float] = [-1.0, 0.0, 1.0],
        max_new_tokens: int = 50,
        **generate_kwargs
    ) -> Dict[float, str]:
        """
        Compare outputs with different steering strengths.
        
        Args:
            prompt: Text prompt to test
            multipliers: List of strength multipliers to test
            max_new_tokens: Maximum new tokens per generation
            **generate_kwargs: Additional generation parameters
            
        Returns:
            Dictionary mapping multipliers to generated outputs
        """
        results = {}
        
        # Ensure vectors are trained
        if any(direction.vector is None for direction in self._directions):
            self.prepare()
        
        original_strengths = [d.strength for d in self._directions]
        
        for multiplier in multipliers:
            # Apply global multiplier
            for direction in self._directions:
                direction.strength = original_strengths[self._directions.index(direction)] * multiplier
            
            # Generate with this multiplier
            result = self.run(
                prompt, 
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
            results[multiplier] = result
        
        # Restore original strengths
        for i, direction in enumerate(self._directions):
            direction.strength = original_strengths[i]
        
        return results
    
    # Conceptor-specific methods for advanced users
    
    def create_conceptor(self, name: str, examples: List[str], alpha: float = 0.1) -> 'Vectsteer':
        """
        Create a conceptor from examples (advanced feature).
        
        Args:
            name: Name of the conceptor
            examples: List of text examples
            alpha: Aperture parameter (default: 0.1)
            
        Returns:
            Self for chaining
        """
        # Record activations for each example
        all_activations = []
        
        for example in examples:
            inputs = self.tokenizer(example, return_tensors="pt").to(self.model.device)
            with record_activations(
                self.model, 
                layer_type=self._layer_type, 
                layer_config=self._enhanced_layer_config,
                layer_nums=self._layer_nums or self._auto_select_layers()
            ) as recorded_activations:
                with torch.no_grad():
                    self.model(**inputs)
                
                # Extract activations for each layer
                for layer_num, acts in recorded_activations.items():
                    # Use the last token's activation
                    all_activations.append(acts[-1][:, -1, :])
        
        # Concatenate all activations
        activations = torch.cat(all_activations, dim=0)
        
        # Compute correlation matrix
        R = (activations.T @ activations) / activations.shape[0]
        
        # Compute conceptor using formula: C = R(R + alpha^(-2)I)^(-1)
        identity = torch.eye(R.shape[0], device=R.device)
        conceptor = R @ torch.inverse(R + (alpha**-2) * identity)
        
        # Store the conceptor
        self._conceptors[name] = conceptor
        return self
    
    def save(self, filepath: str) -> None:
        """
        Save trained steering vectors and conceptors.
        
        Args:
            filepath: Path to save the configuration
        """
        save_data = {
            "directions": [
                {
                    "name": d.name,
                    "strength": d.strength,
                    "read_token_index": d.read_token_index,
                    "layer_activations": {k: v.cpu() for k, v in d.vector.layer_activations.items()} if d.vector else None,
                    "layer_type": d.vector.layer_type if d.vector else None
                }
                for d in self._directions
            ],
            "conceptors": {name: c.cpu() for name, c in self._conceptors.items()},
            "layer_type": self._layer_type,
            "layer_nums": self._layer_nums,
            "aggregator_type": self._aggregator_type.value,
            "operator_type": self._operator_type.value,
            "alpha": self._alpha
        }
        
        torch.save(save_data, filepath)
    
    @staticmethod
    def load(
        filepath: str, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase
    ) -> 'Vectsteer':
        """
        Load saved steering configuration.
        
        Args:
            filepath: Path to saved configuration
            model: Model to apply steering to
            tokenizer: Tokenizer for the model
            
        Returns:
            Initialized Vectsteer object with loaded vectors
        """
        save_data = torch.load(filepath, map_location=model.device)
        
        steerer = Vectsteer(model, tokenizer)
        steerer._layer_type = save_data["layer_type"]
        steerer._layer_nums = save_data["layer_nums"]
        steerer._aggregator_type = AggregatorType(save_data["aggregator_type"])
        steerer._operator_type = OperatorType(save_data["operator_type"])
        steerer._alpha = save_data["alpha"]
        
        # Load directions and vectors
        for dir_data in save_data["directions"]:
            direction = SteeringDirection(
                name=dir_data["name"],
                examples=[],  # Examples not needed after training
                strength=dir_data["strength"],
                read_token_index=dir_data["read_token_index"]
            )
            
            if dir_data["layer_activations"]:
                # Create SteeringVector from saved activations
                layer_activations = {
                    k: v.to(model.device) 
                    for k, v in dir_data["layer_activations"].items()
                }
                
                direction.vector = SteeringVector(
                    layer_activations=layer_activations,
                    layer_type=dir_data["layer_type"]
                )
            
            steerer._directions.append(direction)
        
        # Load conceptors
        steerer._conceptors = {
            name: c.to(model.device) 
            for name, c in save_data["conceptors"].items()
        }
        
        return steerer


# Convenience functions for common use cases

def quick_sentiment_steering(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase,
    strength: float = 1.0
) -> Vectsteer:
    """Quick setup for sentiment steering."""
    examples = [
        ("I absolutely love this and find it amazing", "I really hate this and find it terrible"),
        ("This is fantastic and exceeded my expectations", "This is awful and disappointed me"),
        ("Excellent quality and highly recommended", "Poor quality and not recommended"),
        ("Outstanding performance that impressed me", "Horrible performance that frustrated me"),
        ("Delightful experience that satisfied me", "Frustrating experience that annoyed me")
    ]
    
    return (Vectsteer(model, tokenizer)
            .want("positive_sentiment", examples=examples, strength=strength)
            .using(layer_type="decoder_block"))


def quick_helpfulness_steering(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase,
    strength: float = 1.0
) -> Vectsteer:
    """Quick setup for helpfulness steering."""
    examples = [
        ("I'd be happy to help you with detailed assistance", "I refuse to help you with this request"),
        ("Let me provide clear and thorough explanations", "I'll give you vague and confusing responses"),
        ("I'll offer useful and practical solutions", "I'll provide useless and impractical advice"),
        ("I'll assist you with patient guidance", "I'll dismiss your questions rudely"),
        ("Let me break this down step by step clearly", "I'll make this more complicated than needed")
    ]
    
    return (Vectsteer(model, tokenizer)
            .want("helpful", examples=examples, strength=strength)
            .using(layer_type="mlp"))


def quick_truthfulness_steering(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase,
    strength: float = 1.0
) -> Vectsteer:
    """Quick setup for truthfulness steering.""" 
    examples = [
        ("The Earth is round and orbits the Sun", "The Earth is flat and stationary"),
        ("Vaccines prevent diseases and save lives", "Vaccines cause autism and are harmful"),
        ("Climate change is caused by human activities", "Climate change is completely natural"),
        ("Evolution explains the diversity of species", "All species were created simultaneously"),
        ("Gravity causes objects to fall toward Earth", "Objects fall because they choose to")
    ]
    
    return (Vectsteer(model, tokenizer)
            .want("truthful", examples=examples, strength=strength)
            .using(layer_type="mlp"))


# Quick demo function for Colab/Jupyter
def demo_steering(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    behavior: str = "sentiment",
    test_prompt: str = "I think this movie is"
):
    """
    Quick demonstration of steering for Colab/Jupyter notebooks.
    
    Args:
        model_name: Hugging Face model name
        behavior: "sentiment", "helpfulness", or "truthfulness"
        test_prompt: Prompt to test steering on
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"🔄 Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"🎯 Setting up {behavior} steering...")
    
    if behavior == "sentiment":
        steerer = quick_sentiment_steering(model, tokenizer)
    elif behavior == "helpfulness":
        steerer = quick_helpfulness_steering(model, tokenizer)
    elif behavior == "truthfulness":
        steerer = quick_truthfulness_steering(model, tokenizer)
    else:
        raise ValueError(f"Unknown behavior: {behavior}")
    
    print(f"🧪 Comparing outputs for: '{test_prompt}'")
    results = steerer.compare(test_prompt, multipliers=[-1.0, 0.0, 1.0])
    
    print("=" * 60)
    for mult, output in results.items():
        status = "🚫 AVOID" if mult < 0 else "✅ WANT" if mult > 0 else "⚪ BASELINE"
        print(f"{status} (×{mult:+.1f}): {output}")
        print("-" * 40)
    
    return steerer, results


if __name__ == "__main__":
    # Example usage
    print("Vectsteer: Advanced Steering Vectors API")
    print("=========================================")
    print("Quick demo available with demo_steering()")
    print("For custom steering, use the Vectsteer class directly.")
