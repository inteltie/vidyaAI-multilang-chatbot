import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.language_detector import LanguageDetector

def test_detection():
    detector = LanguageDetector()
    
    test_cases = [
        ("Hello, how are you today?", "en"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
        ("Bonjour, comment ça va ?", "fr"),
        ("¿Cómo estás?", "es"),
        ("你好，你今天怎么样？", "zh"),
    ]
    
    print("Starting language detection tests...")
    print("-" * 40)
    
    all_passed = True
    for text, expected in test_cases:
        detected = detector.detect_language(text)
        status = "PASS" if detected == expected else "FAIL"
        print(f"Text: {text}")
        print(f"Expected: {expected} | Detected: {detected} | Status: {status}")
        if detected != expected:
            all_passed = False
        print("-" * 40)
    
    if all_passed:
        print("All basic detection tests PASSED!")
    else:
        print("Some tests FAILED. Please check language codes.")

if __name__ == "__main__":
    test_detection()
