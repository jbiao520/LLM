## ADDED Requirements

### Requirement: Beginner guide explains attention efficiency
The beginner guide SHALL explain attention variants using analogies:
- Standard attention as "everyone talks to everyone"
- GQA as "groups share the same questions"
- MQA as "everyone uses the same questions"
- Flash Attention as "smart memory management"
- Sliding window as "only looking at nearby context"

#### Scenario: Beginner understands efficiency trade-offs
- **WHEN** a beginner reads the guide
- **THEN** they can explain why GQA saves memory
- **AND** they understand when to use different attention types

### Requirement: Advanced guide covers attention variants mathematically
The advanced guide SHALL provide mathematical foundations:
- Multi-Head Attention (MHA) baseline
- Multi-Query Attention (MQA) - single KV head
- Grouped Query Attention (GQA) - grouped KV heads
- Flash Attention algorithm (memory-efficient attention)
- Sliding Window Attention mechanism
- KV Cache optimization

#### Scenario: Advanced reader implements GQA
- **WHEN** an advanced reader studies the guide
- **THEN** they can implement GQA from MHA code
- **AND** they understand the memory/compute trade-off

### Requirement: Examples demonstrate attention variants
The examples directory SHALL contain runnable Python code demonstrating:
- GQA implementation
- MQA implementation
- Sliding window attention
- KV Cache usage
- Memory comparison between variants

#### Scenario: Learner runs attention example
- **WHEN** learner runs `python examples/gqa_example.py`
- **THEN** they see memory usage comparison
- **AND** they understand the implementation differences

### Requirement: Diagram visualizes attention variants
The diagram file SHALL visualize:
- MHA vs MQA vs GQA head configurations
- Flash Attention memory access pattern
- Sliding window attention mask
- KV Cache structure

#### Scenario: Visual learner understands attention variants
- **WHEN** visual learner views the diagram
- **THEN** they understand how attention heads are grouped
- **AND** they see the memory savings from variants
