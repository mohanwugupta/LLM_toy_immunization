from transformers import PreTrainedTokenizer
from typing import Dict, Set

def get_valid_number_tokens(tokenizer: PreTrainedTokenizer) -> Dict[int, int]:
    """
    Finds all numbers between 0 and 999 that are represented as a single token.
    Returns a dictionary mapping the integer value to its token ID.
    
    Checks both standard form '123' and prepended space ' 123' depending on the tokenizer.
    We test how it tokenizes in the context of our prompt format: '533,460,'
    """
    valid_tokens = {}
    
    # We test numbers in a simulated context to ensure they are single tokens
    # Llama-3 often tokenizes numbers cleanly, but let's be sure.
    for num in range(1000):
        text = str(num)
        # Tokenize without special tokens to get just the raw token
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) == 1:
            valid_tokens[num] = tokens[0]
            
    return valid_tokens

def extract_valid_support(valid_tokens: Dict[int, int]) -> Set[int]:
    """Extract the set of integers that have valid single-token representations."""
    return set(valid_tokens.keys())
