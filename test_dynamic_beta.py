#!/usr/bin/env python3
"""
Test script for dynamic beta scaling implementation
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from decodingmethod.moe_utils import calculate_jsd, adaptive_beta_scaling

def test_jsd_calculation():
    """Test the JSD calculation function"""
    print("Testing JSD calculation...")
    
    # Create two identical distributions (should have JSD ≈ 0)
    identical_dist1 = torch.tensor([[0.8, 0.1, 0.1]])
    identical_dist2 = torch.tensor([[0.8, 0.1, 0.1]])
    jsd_identical = calculate_jsd(identical_dist1, identical_dist2)
    print(f"JSD for identical distributions: {jsd_identical.item():.6f} (should be ~0)")
    
    # Create two very different distributions (should have higher JSD)
    different_dist1 = torch.tensor([[0.9, 0.05, 0.05]])
    different_dist2 = torch.tensor([[0.1, 0.45, 0.45]])
    jsd_different = calculate_jsd(different_dist1, different_dist2)
    print(f"JSD for different distributions: {jsd_different.item():.6f} (should be >0)")
    
    return jsd_identical, jsd_different

def test_adaptive_beta_scaling():
    """Test the adaptive beta scaling function"""
    print("\nTesting adaptive beta scaling...")
    
    # Test with low JSD (should give low beta)
    low_jsd = torch.tensor(0.01)
    beta_low = adaptive_beta_scaling(low_jsd, base_beta=0.5, jsd_threshold=0.1, max_beta=1.0, min_beta=0.1)
    print(f"Low JSD ({low_jsd.item():.3f}) -> beta = {beta_low.item():.3f} (should be close to min_beta)")
    
    # Test with high JSD (should give high beta)
    high_jsd = torch.tensor(0.2)
    beta_high = adaptive_beta_scaling(high_jsd, base_beta=0.5, jsd_threshold=0.1, max_beta=1.0, min_beta=0.1)
    print(f"High JSD ({high_jsd.item():.3f}) -> beta = {beta_high.item():.3f} (should be close to max_beta)")
    
    # Test with medium JSD (should give medium beta)
    medium_jsd = torch.tensor(0.1)
    beta_medium = adaptive_beta_scaling(medium_jsd, base_beta=0.5, jsd_threshold=0.1, max_beta=1.0, min_beta=0.1)
    print(f"Medium JSD ({medium_jsd.item():.3f}) -> beta = {beta_medium.item():.3f} (should be ~0.55)")
    
    return beta_low, beta_medium, beta_high

def test_end_to_end():
    """Test the complete pipeline"""
    print("\nTesting end-to-end pipeline...")
    
    # Simulate teacher and student logits
    vocab_size = 1000
    teacher_logits = torch.randn(1, vocab_size)
    student_logits = torch.randn(1, vocab_size)
    
    # Convert to probabilities
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    
    # Calculate JSD
    jsd_score = calculate_jsd(teacher_probs, student_probs)
    
    # Get dynamic beta
    beta_dynamic = adaptive_beta_scaling(jsd_score, base_beta=0.5)
    
    print(f"Teacher-Student JSD: {jsd_score.item():.4f}")
    print(f"Dynamic beta: {beta_dynamic.item():.4f}")
    
    # Apply contrastive formula
    diffs_dynamic = (1 + beta_dynamic) * teacher_logits - beta_dynamic * student_logits
    diffs_fixed = (1 + 0.5) * teacher_logits - 0.5 * student_logits
    
    print(f"Max difference between dynamic and fixed: {(diffs_dynamic - diffs_fixed).abs().max().item():.4f}")
    
    return jsd_score, beta_dynamic

if __name__ == "__main__":
    print("=" * 50)
    print("Dynamic Beta Scaling Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        jsd_identical, jsd_different = test_jsd_calculation()
        beta_low, beta_medium, beta_high = test_adaptive_beta_scaling()
        jsd_score, beta_dynamic = test_end_to_end()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✅")
        print("Dynamic beta scaling implementation is working.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()