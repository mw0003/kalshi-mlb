#!/usr/bin/env python3
"""
Simple tennis name matching test - standalone version
"""

def normalize_tennis_player_name(full_name):
    """
    Normalize tennis player name to match Kalshi format
    Tries to extract last name and convert to uppercase
    Returns None if normalization fails - implements try-match-ignore-mismatch logic
    """
    if not full_name or not isinstance(full_name, str):
        print(f"⚠️ Invalid tennis player name: {full_name}")
        return None
    
    try:
        name = full_name.strip()
        
        if "," in name:
            last_name = name.split(",")[0].strip()
            print(f"🎾 Tennis name format detected: Last,First -> '{last_name}'")
        elif ". " in name:
            parts = name.split(". ")
            last_name = parts[-1].strip() if len(parts) > 1 else name
            print(f"🎾 Tennis name format detected: Initial.Last -> '{last_name}'")
        else:
            parts = name.split()
            last_name = parts[-1] if parts else name
            print(f"🎾 Tennis name format detected: First Last -> '{last_name}'")
        
        normalized = last_name.upper()
        
        char_map = {
            'Ć': 'C', 'Č': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'D',
            'Ñ': 'N', 'Ü': 'U', 'Ö': 'O', 'Ä': 'A', 'É': 'E',
            'È': 'E', 'À': 'A', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U'
        }
        
        for original, replacement in char_map.items():
            normalized = normalized.replace(original, replacement)
        
        print(f"🎾 Tennis name normalized: '{full_name}' → '{normalized}'")
        return normalized
        
    except Exception as e:
        print(f"❌ Error normalizing tennis name '{full_name}': {e} - ignoring mismatch")
        return None

def test_tennis_name_normalization():
    """Test tennis name normalization with various formats"""
    print("🧪 Testing tennis name normalization...")
    
    test_cases = [
        ("Iga Swiatek", "SWIATEK"),
        ("N. Djokovic", "DJOKOVIC"),
        ("Sabalenka, Aryna", "SABALENKA"),
        ("Carlos Alcaraz", "ALCARAZ"),
        ("Novak Đoković", "DJOKOVIC"),  # International character
        ("", None),  # Empty string
        (None, None),  # None input
        ("Rafael Nadal", "NADAL"),
        ("J. Sinner", "SINNER")
    ]
    
    passed = 0
    failed = 0
    
    for input_name, expected in test_cases:
        result = normalize_tennis_player_name(input_name)
        if result == expected:
            print(f"✅ '{input_name}' → '{result}' (expected: '{expected}')")
            passed += 1
        else:
            print(f"❌ '{input_name}' → '{result}' (expected: '{expected}')")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    print("🎾 TENNIS NAME MATCHING TEST")
    print("=" * 40)
    
    success = test_tennis_name_normalization()
    
    if success:
        print("🎉 All tennis name matching tests passed!")
        print("\n✅ Tennis name matching feature successfully implemented:")
        print("   - Handles various name formats (First Last, Last,First, Initial.Last)")
        print("   - Converts to uppercase for Kalshi matching")
        print("   - Handles international characters with transliteration")
        print("   - Gracefully ignores mismatches as requested")
    else:
        print("💥 Some tests failed!")
