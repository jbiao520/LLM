## ADDED Requirements

### Requirement: MoE Workflow Diagram
The MoE topic SHALL include a diagram.md file showing the Mixture of Experts workflow.

#### Scenario: Reader visualizes MoE routing
- **WHEN** reader views topics/moe/diagram.md
- **THEN** they see a Mermaid flowchart showing router → expert selection → weighted combination flow
