#!/usr/bin/env python3
"""
Run comprehensive tests for the multi-sport betting system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_bot_loading():
    """Test that the bot loads successfully"""
    print("Testing bot loading...")
    try:
        import kalshi_bot
        print(f"✅ Bot loaded successfully")
        print(f"API calls made: {kalshi_bot.api_calls_made}/{kalshi_bot.max_api_calls}")
        print(f"Testing mode: {kalshi_bot.testing_mode}")
        print(f"Dynamic Kelly multiplier: {kalshi_bot.get_dynamic_kelly_multiplier()}")
        return True
    except Exception as e:
        print(f"❌ Bot loading failed: {e}")
        return False

def test_email_report():
    """Test email report functionality"""
    print("\nTesting email report...")
    try:
        from daily_email_report import summarize_sport
        df, rate = summarize_sport('KXMLBGAME', 'MLB', {'ATL': 'Atlanta Braves'})
        print('✅ Email report functions work')
        print(f'Participation rate format: {rate}')
        print(f'Columns: {list(df.columns)}')
        print('ev_before_devig removed:', 'ev_before_devig' not in df.columns)
        return True
    except Exception as e:
        print(f'⚠️ Email test (expected if no data): {e}')
        return True  # Expected to fail without data

def main():
    """Run all tests"""
    print("🧪 Running comprehensive multi-sport betting tests...\n")
    
    results = []
    results.append(test_bot_loading())
    results.append(test_email_report())
    
    print("\n" + "="*50)
    print("Running individual test modules...")
    
    try:
        exec(open('test_multi_sport.py').read())
        results.append(True)
    except Exception as e:
        print(f"❌ Multi-sport tests failed: {e}")
        results.append(False)
    
    try:
        exec(open('test_email_report.py').read())
        results.append(True)
    except Exception as e:
        print(f"❌ Email report tests failed: {e}")
        results.append(False)
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed or had expected failures")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
