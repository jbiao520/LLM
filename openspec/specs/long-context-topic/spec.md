## ADDED Requirements

### Requirement: Beginner guide explains long context challenges
The beginner guide SHALL explain long context using analogies:
- Context window as "short-term memory limit"
- Why models struggle with long documents
- RoPE scaling as "stretching the position map"
- Why longer context is valuable

#### Scenario: Beginner understands context limits
- **WHEN** a beginner reads the guide
- **THEN** they can explain why models have context length limits
- **AND** they understand the trade-off between context and compute

### Requirement: Advanced guide covers context extension techniques
The advanced guide SHALL provide mathematical foundations:
- RoPE (Rotary Position Embedding) fundamentals
- Linear RoPE scaling
- Dynamic NTK-aware scaling
- YaRN (Yet another RoPE extensioN)
- ALiBi (Attention with Linear Biases)
- LongLoRA approach
- Context length extrapolation theory

#### Scenario: Advanced reader implements RoPE scaling
- **WHEN** an advanced reader studies the guide
- **THEN** they can implement linear RoPE scaling
- **AND** they understand why certain scaling factors work better

### Requirement: Examples demonstrate context extension
The examples directory SHALL contain runnable Python code demonstrating:
- RoPE scaling implementation
- Different scaling strategies comparison
- Attention pattern changes with scaling

#### Scenario: Learner runs scaling example
- **WHEN** learner runs `python examples/rope_scaling_example.py`
- **THEN** they see how scaling affects position embeddings
- **AND** they can compare different scaling methods

### Requirement: Diagram visualizes context extension
The diagram file SHALL visualize:
- Original context window limitation
- RoPE scaling effect on position embeddings
- Different scaling strategies comparison
- Attention pattern with extended context

#### Scenario: Visual learner understands extension
- **WHEN** visual learner views the diagram
- **THEN** they understand how scaling extends context
- **AND** they see the trade-offs of different methods
