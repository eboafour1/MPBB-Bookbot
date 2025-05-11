def chunk_text(text: str, max_words: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
