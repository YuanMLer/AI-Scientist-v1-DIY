#!/usr/bin/env python3
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_scientist.llm import (
    create_client,
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
    AVAILABLE_LLMS,
    OLLAMA_MODELS
)

def test_imports():
    print("=" * 60)
    print("Test 1: Import Verification")
    print("=" * 60)
    
    print(f"OLLAMA_MODELS: {OLLAMA_MODELS}")
    print(f"AVAILABLE_LLMS: {AVAILABLE_LLMS}")
    print(f"OLLAMA_MODELS is AVAILABLE_LLMS: {OLLAMA_MODELS is AVAILABLE_LLMS}")
    print("✓ Test passed\n")
    return True

def test_backward_compatibility():
    print("=" * 60)
    print("Test 2: Backward Compatibility")
    print("=" * 60)
    
    assert AVAILABLE_LLMS == OLLAMA_MODELS, "AVAILABLE_LLMS should equal OLLAMA_MODELS"
    assert len(AVAILABLE_LLMS) == 3, "Should have 3 models"
    assert "qwen3-next:latest" in AVAILABLE_LLMS, "qwen3-next:latest should be available"
    print("✓ Test passed\n")
    return True

def test_client_creation():
    print("=" * 60)
    print("Test 3: Client Creation")
    print("=" * 60)
    
    ollama_base_url = "http://1.13.248.121:17719"
    os.environ["OLLAMA_BASE_URL"] = ollama_base_url
    
    for model in OLLAMA_MODELS:
        try:
            client, actual_model = create_client(model)
            print(f"✓ Created client for {model}")
        except Exception as e:
            print(f"✗ Failed to create client for {model}: {e}")
            return False
    
    print("✓ Test passed\n")
    return True

def test_single_response():
    print("=" * 60)
    print("Test 4: Single Response")
    print("=" * 60)
    
    model = "qwen3-next:latest"
    ollama_base_url = "http://1.13.248.121:17719"
    os.environ["OLLAMA_BASE_URL"] = ollama_base_url
    
    try:
        client, actual_model = create_client(model)
        response, history = get_response_from_llm(
            msg="请用一句话介绍Python。",
            client=client,
            model=actual_model,
            system_message="你是一个助手。",
            print_debug=False,
            temperature=0.7
        )
        print(f"Response: {response}")
        print(f"History length: {len(history)}")
        print("✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        traceback.print_exc()
        return False

def test_batch_response():
    print("=" * 60)
    print("Test 5: Batch Response")
    print("=" * 60)
    
    model = "qwen3-next:latest"
    ollama_base_url = "http://1.13.248.121:17719"
    os.environ["OLLAMA_BASE_URL"] = ollama_base_url
    
    try:
        client, actual_model = create_client(model)
        responses, histories = get_batch_responses_from_llm(
            msg="请列举Python的两个优点。",
            client=client,
            model=actual_model,
            system_message="你是一个助手。",
            print_debug=False,
            temperature=0.7,
            n_responses=2
        )
        print(f"Number of responses: {len(responses)}")
        for i, resp in enumerate(responses):
            print(f"Response {i+1}: {resp[:100]}...")
        print("✓ Test passed\n")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}\n")
        traceback.print_exc()
        return False

def test_json_extraction():
    print("=" * 60)
    print("Test 6: JSON Extraction")
    print("=" * 60)
    
    test_cases = [
        ("```json{\"key\": \"value\"}```", {"key": "value"}),
        ("Text {\"a\": 1, \"b\": 2} more text", {"a": 1, "b": 2}),
        ("No JSON", None),
    ]
    
    for i, (input_text, expected) in enumerate(test_cases):
        result = extract_json_between_markers(input_text)
        if expected is None and result is None:
            print(f"Case {i+1}: ✓ Correctly returned None")
        elif result == expected:
            print(f"Case {i+1}: ✓ Extracted: {result}")
        else:
            print(f"Case {i+1}: ✗ Expected {expected}, got {result}")
            return False
    
    print("✓ Test passed\n")
    return True

def test_invalid_model():
    print("=" * 60)
    print("Test 7: Invalid Model Handling")
    print("=" * 60)
    
    try:
        create_client("invalid-model")
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ Unexpected exception: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("SIMPLIFIED OLLAMA LLM INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_backward_compatibility():
        all_passed = False
    
    if not test_client_creation():
        all_passed = False
    
    if not test_single_response():
        all_passed = False
    
    if not test_batch_response():
        all_passed = False
    
    if not test_json_extraction():
        all_passed = False
    
    if not test_invalid_model():
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
