import re

COMMON_SYMBOLS = {
    "RELIANCE",
    "HDFC",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "TCS",
    "INFY",
    "LT",
    "AXISBANK",
    "KOTAKBANK",
    "BAJFINANCE",
    "TATAMOTORS",
    "ADANIENT",
    "ADANIPORTS",
}


def extract_symbols(text: str) -> set[str]:
    tokens = set(re.findall(r"\b[A-Z]{2,15}\b", text.upper()))
    return {token for token in tokens if token in COMMON_SYMBOLS}
