def bias_check(summary: str) -> str:
    blacklist = ["controversial_term1", "controversial_term2"]
    for term in blacklist:
        summary = summary.replace(term, "[REDACTED]")
    return summary