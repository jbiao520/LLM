## ADDED Requirements

### Requirement: Mermaid Diagram Format
All topic diagrams SHALL use Mermaid syntax for portability and GitHub compatibility.

#### Scenario: Diagram renders on GitHub
- **WHEN** diagram.md is viewed on GitHub
- **THEN** the Mermaid diagram renders correctly as a visual flowchart

### Requirement: Diagram Completeness
Each diagram SHALL show the complete workflow of its topic, including inputs, processing steps, and outputs.

#### Scenario: Diagram shows full workflow
- **WHEN** reader views any topic's diagram.md
- **THEN** they see all major components and their relationships

### Requirement: Diagram Accessibility
Each diagram SHALL include text explanations accompanying the visual representation.

#### Scenario: Reader understands diagram without visuals
- **WHEN** Mermaid rendering is unavailable
- **THEN** the text explanations still convey the workflow clearly

### Requirement: Diagram Consistency
All topic diagrams SHALL follow a consistent visual style and notation.

#### Scenario: Diagrams have uniform style
- **WHEN** reader views multiple topic diagrams
- **THEN** they see consistent use of shapes, colors, and labels
