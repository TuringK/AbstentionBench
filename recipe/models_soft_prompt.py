"""
Soft Prompt Model for AbstentionBench Evaluation
================================================
Loads a base model with a trained PEFT soft prompt adapter.

Add to recipe/models/__init__.py:
    from recipe.models.soft_prompt_gemma import Gemma_3_1B_SoftPrompt
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from recipe.models import InferenceModel

logger = logging.getLogger(__name__)


class SoftPromptModelBase(InferenceModel):
    """
    Base class for models with soft prompt adapters.
    Uses HuggingFace Transformers (not vLLM) for PEFT compatibility.
    """
    
    def __init__(
        self,
        base_model_path: str,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        tensor_parallel_size: int = 1,  # For AbstentionBench compatibility (not used)
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size  # Store for compatibility
        
        logger.info(f"Loading base model: {base_model_path}")
        logger.info(f"Loading soft prompt from: {soft_prompt_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(self.model, soft_prompt_path)
        self.model.eval()
        
        self.device = next(self.model.parameters()).device
        logger.info(f"Model loaded on device: {self.device}")
    
    def question_to_chat_format(self, question: str) -> str:
        """Convert question to chat template format."""
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def respond(self, questions: list[str]) -> list[str]:
        """Generate responses for a batch of questions."""
        if isinstance(questions, str):
            logger.info("Wrapping string question into a list.")
            questions = [questions]
        
        responses = []
        
        # Process one at a time to avoid OOM with long sequences
        # For 180 eval samples, this is fast enough
        for question in questions:
            formatted = self.question_to_chat_format(question)
            
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode and remove the input prompt from output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the input)
            input_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            response = full_response[len(input_text):].strip()
            
            responses.append(response)
        
        logger.info(f"Sample responses: {responses[:3]}")
        return responses


class Gemma_3_1B_SoftPrompt(SoftPromptModelBase):
    """Gemma 3 1B with trained soft prompt for abstention."""
    
    def __init__(
        self,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        tensor_parallel_size: int = 1,  # For AbstentionBench compatibility
    ):
        super().__init__(
            base_model_path="google/gemma-3-1b-it",
            soft_prompt_path=soft_prompt_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )


# ============================================================
# Additional model classes for other base models
# ============================================================

class OLMo_7B_SoftPrompt(SoftPromptModelBase):
    """OLMo 7B with trained soft prompt for abstention."""
    
    def __init__(
        self,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(
            base_model_path="allenai/OLMo-7B-0724-Instruct-hf",
            soft_prompt_path=soft_prompt_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )


class Qwen2_5_1_5B_SoftPrompt(SoftPromptModelBase):
    """Qwen 2.5 1.5B with trained soft prompt for abstention."""
    
    def __init__(
        self,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(
            base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            soft_prompt_path=soft_prompt_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )


class Llama3_1_8B_SoftPrompt(SoftPromptModelBase):
    """Llama 3.1 8B with trained soft prompt for abstention."""
    
    def __init__(
        self,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(
            base_model_path="meta-llama/Llama-3.1-8B-Instruct",
            soft_prompt_path=soft_prompt_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )

class OLMo_3_7B_SoftPrompt(SoftPromptModelBase):
    """OLMo 3 7B Instruct with trained soft prompt for abstention."""
    
    def __init__(
        self,
        soft_prompt_path: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(
            base_model_path="allenai/Olmo-3-7B-Instruct",
            soft_prompt_path=soft_prompt_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
        )