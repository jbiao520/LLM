## ADDED Requirements

### Requirement: Full Fine-tuning Beginner Guide
The system SHALL provide a beginner-friendly guide explaining full parameter fine-tuning concepts.

#### Scenario: Zero-background reader understands full fine-tuning
- **WHEN** a reader with no ML background reads beginner-guide.md
- **THEN** they understand what full fine-tuning is and when to use it vs LoRA

### Requirement: Full Fine-tuning Advanced Guide
The system SHALL provide an advanced guide covering training dynamics, learning rate schedules, and optimization.
It SHALL include detailed Chinese explanations for every mathematical formula, including variable definitions and intuition, to support readers with weak math background.

#### Scenario: ML practitioner understands full fine-tuning details
- **WHEN** a reader with ML background reads advanced-guide.md
- **THEN** they understand training hyperparameters, gradient accumulation, and mixed precision training

#### Scenario: Formulas are explained in Chinese
- **WHEN** a reader with ML background reads advanced-guide.md
- **THEN** the guide explains every mathematical formula in Chinese, including variable definitions and intuition

### Requirement: Full Fine-tuning Code Examples
The system SHALL provide runnable Python code examples demonstrating full model fine-tuning.

#### Scenario: Developer runs full fine-tuning example
- **WHEN** developer runs full_finetune_example.py
- **THEN** they see a working full fine-tuning example with training loop

### Requirement: Full Fine-tuning Workflow Diagram
The system SHALL provide a visual diagram showing the full fine-tuning workflow.

#### Scenario: Reader visualizes full fine-tuning process
- **WHEN** reader views diagram.md
- **THEN** they see a clear flowchart showing data preparation, training, and evaluation steps
