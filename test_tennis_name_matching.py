#!/usr/bin/env python3
"""
Test tennis name matching functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tennis_name_normalization():
    """Test tennis name normalization with various formats"""
    print("🧪 Testing tennis name normalization...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kalshi_bot", "kalshi_bot.py")
        kalshi_module = importlib.util.module_from_spec(spec)
        
        import sys
        sys.modules['requests'] = type(sys)('requests')
        sys.modules['requests'].get = lambda *args, **kwargs: None
        
        spec.loader.exec_module(kalshi_module)
        normalize_tennis_player_name = kalshi_module.normalize_tennis_player_name
        
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
            result = normalize_tennis_player_name_test(input_name)
            if result == expected:
                print(f"✅ '{input_name}' → '{result}' (expected: '{expected}')")
                passed += 1
            else:
                print(f"❌ '{input_name}' → '{result}' (expected: '{expected}')")
                failed += 1
        
        print(f"\n📊 Test Results: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("🎾 TENNIS NAME MATCHING TEST")
    print("=" * 40)
    
    success = test_tennis_name_normalization()
    
    if success:
        print("🎉 All tennis name matching tests passed!")
    else:
        print("💥 Some tests failed!")
        sys.exit(1)
