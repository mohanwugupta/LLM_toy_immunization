import pytest
from src.data.tokenize_numbers import get_valid_number_tokens, extract_valid_support

class DummyTokenizer:
    """A dummy tokenizer for testing tokenization logic."""
    def encode(self, text, add_special_tokens=False):
        if text == "123":
            return [10123] # Single token
        elif text == "45":
            return [1004, 1005] # Two tokens
        else:
            # Let's say all other numbers < 10 are single tokens
            if text.isdigit() and int(text) < 10:
                return [2000 + int(text)]
            return [1000, 1001]

def test_number_token_extraction():
    """Number-token extraction should not include non-number tokens"""
    tokenizer = DummyTokenizer()
    valid_tokens = get_valid_number_tokens(tokenizer)
    
    # "123" should be valid
    assert 123 in valid_tokens
    assert valid_tokens[123] == 10123
    
    # "45" should not be valid
    assert 45 not in valid_tokens

def test_valid_single_tokens():
    """Evaluation should only score numbers that are valid single tokens"""
    tokenizer = DummyTokenizer()
    valid_tokens = get_valid_number_tokens(tokenizer)
    support = extract_valid_support(valid_tokens)
    
    assert 123 in support
    assert 45 not in support
    for i in range(10):
        assert i in support
