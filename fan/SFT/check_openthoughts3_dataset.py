#!/usr/bin/env python3
"""
Dataset quality check script for OpenThoughts3-1.2M
This script validates the dataset format and checks for potential issues.
"""

import argparse
from datasets import load_dataset


def check_openthoughts3_dataset(dataset_name: str, num_samples: int = 10):
    """
    Check the quality of OpenThoughts3 dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to check (0 = all)
    """
    print("=" * 80)
    print("OpenThoughts3 Dataset Quality Check")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    
    # Statistics
    total_samples = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))
    invalid_samples = 0
    issues = {
        "missing_conversations": 0,
        "empty_conversations": 0,
        "invalid_message_format": 0,
        "empty_content": 0,
        "unknown_role": 0,
    }
    role_stats = {}
    
    print(f"\nChecking {total_samples} samples...")
    
    for i in range(total_samples):
        sample = dataset[i]
        
        # Check conversations field
        if "conversations" not in sample:
            issues["missing_conversations"] += 1
            if issues["missing_conversations"] <= 5:
                print(f"  [Sample {i}] Missing 'conversations' field")
            invalid_samples += 1
            continue
        
        conversations = sample["conversations"]
        
        # Check if conversations is non-empty
        if not isinstance(conversations, list) or len(conversations) == 0:
            issues["empty_conversations"] += 1
            if issues["empty_conversations"] <= 5:
                print(f"  [Sample {i}] Empty or invalid 'conversations'")
            invalid_samples += 1
            continue
        
        # Check each message in conversations
        sample_valid = True
        for j, msg in enumerate(conversations):
            # Check message format
            if not isinstance(msg, dict):
                issues["invalid_message_format"] += 1
                if issues["invalid_message_format"] <= 5:
                    print(f"  [Sample {i}][Msg {j}] Not a dict: {type(msg)}")
                sample_valid = False
                break
            
            # Check required fields
            if "from" not in msg or "value" not in msg:
                issues["invalid_message_format"] += 1
                if issues["invalid_message_format"] <= 5:
                    print(f"  [Sample {i}][Msg {j}] Missing 'from' or 'value': {msg.keys()}")
                sample_valid = False
                break
            
            # Check content
            if not msg["value"] or not msg["value"].strip():
                issues["empty_content"] += 1
                if issues["empty_content"] <= 5:
                    print(f"  [Sample {i}][Msg {j}] Empty content")
                sample_valid = False
                break
            
            # Track role statistics
            role = msg["from"]
            role_stats[role] = role_stats.get(role, 0) + 1
            
            # Check for unknown roles (should be 'human' or 'gpt')
            if role not in ["human", "gpt", "system", "user", "assistant"]:
                issues["unknown_role"] += 1
                if issues["unknown_role"] <= 5:
                    print(f"  [Sample {i}][Msg {j}] Unknown role: {role}")
        
        if not sample_valid:
            invalid_samples += 1
    
    # Print results
    print("\n" + "=" * 80)
    print("Quality Check Results")
    print("=" * 80)
    print(f"Total samples checked: {total_samples}")
    print(f"Valid samples: {total_samples - invalid_samples}")
    print(f"Invalid samples: {invalid_samples}")
    print(f"Valid rate: {(total_samples - invalid_samples) / total_samples * 100:.2f}%")
    
    print("\nIssue breakdown:")
    for issue_type, count in issues.items():
        if count > 0:
            print(f"  {issue_type}: {count}")
    
    print("\nRole statistics:")
    for role, count in sorted(role_stats.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count}")
    
    # Show sample data
    print("\n" + "=" * 80)
    print("Sample Data Examples")
    print("=" * 80)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i+1} ---")
        print(f"Source: {sample.get('source', 'N/A')}")
        print(f"Domain: {sample.get('domain', 'N/A')}")
        print(f"Difficulty: {sample.get('difficulty', 'N/A')}")
        print(f"Conversations ({len(sample['conversations'])} messages):")
        for j, msg in enumerate(sample['conversations']):
            role = msg['from']
            content = msg['value']
            preview = content[:150] + "..." if len(content) > 150 else content
            print(f"  [{j}] {role}: {preview}")
    
    print("\n" + "=" * 80)
    
    if invalid_samples == 0:
        print("✓ All checked samples are valid!")
        return True
    else:
        print(f"✗ Found {invalid_samples} invalid samples")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check OpenThoughts3 dataset quality")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="open-thoughts/OpenThoughts3-1.2M",
        help="Dataset name or path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to check (0 = all)",
    )
    
    args = parser.parse_args()
    check_openthoughts3_dataset(args.dataset_name, args.num_samples)
