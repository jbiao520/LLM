## ADDED Requirements

### Requirement: Beginner guide explains tokenization conceptually
The beginner guide SHALL explain tokenization using intuitive analogies, covering:
- Why text must be converted to numbers for LLMs
- What is a token (subword, character, word levels)
- How tokenization affects model behavior and costs
- Common tokenization artifacts (spacing issues, rare words)

#### Scenario: Beginner understands tokenization basics
- **WHEN** a beginner reads the guide
- **THEN** they can explain why "hello world" becomes multiple tokens
- **AND** they understand why some words are split strangely

### Requirement: Advanced guide covers BPE algorithm mathematically
The advanced guide SHALL provide mathematical explanation of:
- Byte Pair Encoding (BPE) algorithm with pseudocode
- Vocabulary construction process
- Encoding and decoding algorithms
- BBPE (Byte-level BPE) differences from standard BPE
- Comparison with WordPiece and Unigram tokenization

#### Scenario: Advanced reader implements BPE
- **WHEN** an advanced reader studies the guide
- **THEN** they can implement a basic BPE tokenizer from scratch
- **AND** they understand the trade-offs between tokenization methods

### Requirement: Examples demonstrate real tokenizer usage
The examples directory SHALL contain runnable Python code demonstrating:
- tiktoken usage (GPT-compatible tokenizer)
- sentencepiece usage (LLaMA-compatible tokenizer)
- BPE algorithm implementation from scratch
- Tokenization edge cases and debugging

#### Scenario: Learner runs tokenizer examples
- **WHEN** learner runs `python examples/tiktoken_example.py`
- **THEN** they see tokens, token IDs, and decoded output
- **AND** they can modify code to tokenize their own text

### Requirement: Diagram visualizes tokenization pipeline
The diagram file SHALL visualize:
- Text to tokens conversion process
- Vocabulary lookup mechanism
- Encoding vs decoding flow
- Special tokens handling (BOS, EOS, PAD)

#### Scenario: Visual learner understands flow
- **WHEN** visual learner views the diagram
- **THEN** they understand the complete tokenization pipeline
- **AND** they can trace text through encode → IDs → decode
