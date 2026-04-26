import re

def extract_reference_title(user_input: str) -> str:
    """Extract a movie title that appears after 'like', 'similar to', or 'reminds me of'."""
    
    # The updated pattern:
    # 1. Finds "like" (excluding "like to"), OR "similar to", OR "reminds me of"
    # 2. Captures the text immediately following it
    # 3. Stops at conjunctions (but, and, etc.) or punctuation
    pattern = r"\b(?:like(?!\s+to\b)|similar to|reminds me of)\s+(.+?)(?=\b(?:but|and|or|because|with|then|now)\b|[.,!?]|$)"
    
    matches = re.findall(pattern, user_input, re.IGNORECASE)
    
    if not matches:
        return ""
        
    title = matches[-1].strip()
    return title
user_input=input()
print(extract_reference_title(user_input))