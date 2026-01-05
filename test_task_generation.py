#!/usr/bin/env python3
"""
Test the execution task generation fix locally.
Verifies that the semantic clue is always included in the context.
"""

import sys
import os

# Add QUASAR-SUBNET to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'QUASAR-SUBNET'))

from quasar.benchmarks.benchmark_loader import ContextualNeedleLoader

def test_clue_inclusion():
    """Test that the semantic clue is always included in generated tasks."""
    
    print("🧪 Testing execution task generation fix...")
    print("=" * 80)
    
    # Create loader
    loader = ContextualNeedleLoader()
    
    # Generate multiple tasks to test consistency
    num_tests = 5
    passed = 0
    failed = 0
    
    for i in range(num_tests):
        print(f"\n📝 Test {i+1}/{num_tests}: Generating task...")
        
        try:
            task = loader.generate_task(n_distractors=4, target_len=32000)
            
            # Check metadata exists
            metadata = task.metadata
            if not metadata:
                print(f"❌ FAIL: No metadata in task")
                failed += 1
                continue
            
            clue = metadata.get('clue')
            if not clue:
                print(f"❌ FAIL: No clue in metadata")
                failed += 1
                continue
            
            # Check clue is in context
            if clue not in task.context:
                print(f"❌ FAIL: Clue not found in context!")
                print(f"   Clue: {clue[:100]}...")
                failed += 1
                continue
            
            # Check all required metadata fields
            required_fields = ['system_name', 'true_mode', 'true_mode_desc', 'divisor', 'target_remainder', 'clue']
            missing_fields = [f for f in required_fields if f not in metadata]
            
            if missing_fields:
                print(f"❌ FAIL: Missing metadata fields: {missing_fields}")
                failed += 1
                continue
            
            # Success!
            print(f"✅ PASS: Task generated correctly")
            print(f"   System: {metadata['system_name']}")
            print(f"   Mode: {metadata['true_mode']} ({metadata['true_mode_desc']})")
            print(f"   Expected: {task.expected_output}")
            print(f"   Context length: {task.context_length}")
            print(f"   Clue in context: YES")
            
            passed += 1
            
        except Exception as e:
            print(f"❌ FAIL: Exception during generation: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed}/{num_tests} passed")
    
    if failed == 0:
        print("✅ All tests passed! The fix is working correctly.")
        return 0
    else:
        print(f"❌ {failed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(test_clue_inclusion())
