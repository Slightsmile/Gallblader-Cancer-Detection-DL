# Knowledge Graphs for Model Interpretability

This document explains the four knowledge graph types added to the notebook and why you might choose each. It also explains data/annotation requirements and recommended next steps.

## Goals
- Improve interpretability and traceability of model decisions
- Provide structured artifacts you can inspect, query and integrate with domain knowledge
- Help debugging, reporting to clinicians, and identifying systematic errors

## Graph types

### 1) Diagnostic / Learning Graph  (low effort)
- Nodes: epoch nodes (per model), summary nodes
- Edges: sequential epoch edges
- Node attributes: train/val loss, train/val accuracy, epoch number, best validation
- Value: shows training dynamics and can help detect convergence issues, catastrophic forgetting, or instability
- Data needed: training histories (loss/acc per epoch). These are cheap to capture.

### 2) Model–Concept Graph (medium effort)
- Nodes: model nodes, cluster nodes (concepts discovered in embedding space), sample image nodes
- Edges: model->cluster, cluster->class (majority label), cluster->sample
- Node attributes: cluster support, majority label and support
- Value: surfaces learned concepts (groups of similar embeddings), ties them to classes and example inputs
- Data needed: embeddings from model feature layers and test/validation samples; then clustering (KMeans/HDBSCAN)

### 3) Semantic / Domain KG (high effort)
- Nodes: domain concepts (clinical findings), labels, risk factors
- Edges: curated relations (causal, associated_with, subtype_of)
- Value: connects model outputs to domain concepts clinicians care about; enables richer explanation
- Data needed: domain curation (expert annotations or mappings to public ontologies)

### 4) Hybrid Graph (integration)
- Combines model-concept graph nodes with semantic nodes
- Edges map model concepts to domain concepts (via automatic heuristics + manual curation)
- Value: best of both — shows what the model represents and maps it to domain meaningal concepts

## Implementation notes (notebook)
- The notebook includes conservative helper functions to:
  - extract model embeddings in a robust way (tries common hooks)
  - cluster embeddings and build cluster nodes
  - create diagnostic/epoch graph from training histories (if available)
  - create a placeholder semantic KG (needs curation)
  - integrate into a hybrid graph and visualize with NetworkX
- Demo runs are disabled by default to avoid heavy computation. To run them, set `RUN_KG_DEMO = True` in the demo cell.

## Trade-offs and recommendations
- Start with the Diagnostic/Model-Concept graphs (fast to compute) and gather immediate insights.
- Build semantic KG only once you have a small controlled set of domain concepts and/or help from a domain expert — this requires curation.
- Use the hybrid graph to produce clinician-facing explanations: map clusters to concepts and show representative images.

## Next steps (suggested roadmap)
1. Run the model-concept demo on a small subsample to check clusters and representative samples.
2. Curate a small semantic dictionary (5–10 high-value concepts) and add edges to the semantic KG.
3. Map clusters to the curated concepts and review with an expert.
4. Create an interactive explorer (Streamlit / Dash) to let clinicians explore the hybrid KG.

If you want, I can:
- Add a small cell to persist graphs to files (GraphML/JSON) and an export utility for neo4j or pyvis.
- Extend the cluster labeling to use TCAV-style concept attribution (needs labeled concept images).

---

Created automatically as part of this project to document knowledge graphs and next steps.
