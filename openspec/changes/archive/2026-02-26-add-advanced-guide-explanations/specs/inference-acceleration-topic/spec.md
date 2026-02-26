## MODIFIED Requirements

### Requirement: Inference Acceleration Advanced Guide
The system SHALL provide an advanced guide covering vLLM, TensorRT-LLM, continuous batching, and KV Cache optimization.
It SHALL include detailed Chinese explanations for every mathematical formula, including variable definitions and intuition, to support readers with weak math background.

#### Scenario: ML practitioner understands acceleration methods
- **WHEN** a reader with ML background reads advanced-guide.md
- **THEN** they understand PagedAttention, continuous batching, and how to optimize throughput

#### Scenario: Formulas are explained in Chinese
- **WHEN** a reader with ML background reads advanced-guide.md
- **THEN** the guide explains every mathematical formula in Chinese, including variable definitions and intuition
