## ADDED Requirements

### Requirement: Beginner guide explains alignment conceptually
The beginner guide SHALL explain alignment using analogies:
- Alignment as "teaching the model to be helpful and safe"
- SFT as "showing the model good examples"
- RLHF as "rewarding good responses"
- DPO as "learning from preferences directly"
- Why raw pre-trained models can be harmful

#### Scenario: Beginner understands alignment purpose
- **WHEN** a beginner reads the guide
- **THEN** they can explain why ChatGPT needs alignment
- **AND** they understand the difference between base and chat models

### Requirement: Advanced guide covers alignment algorithms mathematically
The advanced guide SHALL provide mathematical foundations:
- SFT loss function and training process
- Reward model training
- PPO algorithm for RLHF
- DPO loss function and derivation
- KL divergence penalty
- Comparison of alignment methods

#### Scenario: Advanced reader implements DPO
- **WHEN** an advanced reader studies the guide
- **THEN** they can implement the DPO loss function
- **AND** they understand why DPO is simpler than PPO

### Requirement: Examples demonstrate alignment concepts
The examples directory SHALL contain runnable Python code demonstrating:
- SFT training loop structure
- Reward model scoring
- DPO loss calculation
- Preference dataset format

#### Scenario: Learner runs alignment example
- **WHEN** learner runs `python examples/dpo_example.py`
- **THEN** they see how preference pairs are used
- **AND** they understand the DPO training process

### Requirement: Diagram visualizes alignment pipeline
The diagram file SHALL visualize:
- SFT training flow
- RLHF pipeline (RM training → PPO)
- DPO direct preference learning
- Comparison of methods

#### Scenario: Visual learner understands alignment flow
- **WHEN** visual learner views the diagram
- **THEN** they understand the complete alignment pipeline
- **AND** they see how human feedback improves the model
