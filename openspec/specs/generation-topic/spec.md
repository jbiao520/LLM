## ADDED Requirements

### Requirement: Beginner guide explains sampling strategies intuitively
The beginner guide SHALL explain text generation using analogies:
- Greedy decoding as "always picking the most likely word"
- Temperature as "creativity dial"
- Top-k as "picking from top k options"
- Top-p (nucleus) as "picking from enough options to reach probability threshold"
- Why different strategies produce different outputs

#### Scenario: Beginner understands sampling trade-offs
- **WHEN** a beginner reads the guide
- **THEN** they can explain why temperature=0 gives deterministic output
- **AND** they understand why high temperature can produce nonsense

### Requirement: Advanced guide covers sampling mathematically
The advanced guide SHALL provide mathematical foundations:
- Softmax and temperature scaling formula
- Top-k filtering algorithm
- Top-p (nucleus) sampling algorithm
- Beam search algorithm with complexity analysis
- Repetition penalty implementation
- Typical sampling and other advanced methods

#### Scenario: Advanced reader implements sampling
- **WHEN** an advanced reader studies the guide
- **THEN** they can implement temperature, top-k, and top-p sampling
- **AND** they understand the probability distributions at each step

### Requirement: Examples demonstrate generation strategies
The examples directory SHALL contain runnable Python code demonstrating:
- All sampling strategies with visual output
- Beam search implementation
- Comparison of outputs across different settings
- Repetition penalty effects

#### Scenario: Learner experiments with generation
- **WHEN** learner runs `python examples/sampling_strategies.py`
- **THEN** they see outputs from greedy, top-k, top-p, and various temperatures
- **AND** they can compare quality and diversity of outputs

### Requirement: Diagram visualizes generation process
The diagram file SHALL visualize:
- Token-by-token generation loop
- How logits become probabilities
- Sampling decision points
- Beam search expansion

#### Scenario: Visual learner understands generation loop
- **WHEN** visual learner views the diagram
- **THEN** they understand how each token is generated
- **AND** they see how sampling affects the generation path
