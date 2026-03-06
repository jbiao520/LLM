## ADDED Requirements

### Requirement: Beginner guide explains pre-training conceptually
The beginner guide SHALL explain pre-training using analogies:
- Pre-training as "learning to predict the next word"
- How reading massive text teaches language understanding
- Difference between pre-training and fine-tuning
- Why pre-trained models have "knowledge"

#### Scenario: Beginner understands pre-training purpose
- **WHEN** a beginner reads the guide
- **THEN** they can explain why pre-training requires massive data
- **AND** they understand that models learn patterns, not facts

### Requirement: Advanced guide covers training objectives mathematically
The advanced guide SHALL provide mathematical foundations:
- Next Token Prediction (autoregressive) loss function
- Masked Language Modeling (MLM) loss function
- Cross-entropy loss for language modeling
- Training data preparation and tokenization
- Scaling laws basics (Chinchilla scaling)

#### Scenario: Advanced reader understands training math
- **WHEN** an advanced reader studies the guide
- **THEN** they can write the cross-entropy loss for next token prediction
- **AND** they understand why perplexity measures model quality

### Requirement: Examples demonstrate training concepts
The examples directory SHALL contain runnable Python code demonstrating:
- Next token prediction on small corpus
- Loss calculation and backpropagation
- Perplexity calculation
- Simple training loop structure

#### Scenario: Learner runs training example
- **WHEN** learner runs `python examples/next_token_prediction.py`
- **THEN** they see loss decreasing over training steps
- **AND** they understand the training loop structure

### Requirement: Diagram visualizes pre-training pipeline
The diagram file SHALL visualize:
- Training data flow (corpus → batches → model → loss)
- Next token prediction objective
- MLM objective (for encoder models)
- Gradient flow and weight updates

#### Scenario: Visual learner understands training flow
- **WHEN** visual learner views the diagram
- **THEN** they understand how data flows through training
- **AND** they see how the model learns from prediction errors
