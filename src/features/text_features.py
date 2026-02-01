def response_length(text: str) -> int:
    return len(text.split())

def refusal_flag(text: str) -> int:
    triggers = [
        "i can't", "i cannot", "i’m unable", "i am unable",
        "i don't have access", "cannot help with"
    ]
    t = text.lower()
    return int(any(x in t for x in triggers))
