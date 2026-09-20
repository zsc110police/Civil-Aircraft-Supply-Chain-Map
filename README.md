# Civil-Aircraft-Supply-Chain-Maps
The data and code are associated with the paper *'A data-driven approach for improving supply chain visibility through knowledge graph construction'.
## Abstract
Supply chains have become increasingly complex due to globalized and fragmented production, making it difficult to understand the structure of complex supply chain systems. This study develops a data-driven framework for reconstructing multi-tier supply chain networks from fragmented textual data using knowledge graph techniques. The framework extracts organizational and product relationships from unstructured text and transforms them into structured representations of supply chain systems. To address incomplete network structures caused by limited recall in information extraction, a rule-based multi-hop reasoning mechanism is introduced to infer implicit supplier relationships from existing organizational and product connections. In addition, a cost-efficient annotation strategy integrating human expertise with large language model-based aggregation is developed to construct high-quality domain-specific training data, addressing the scarcity of labeled data in specialized industrial settings. The framework is evaluated through a civil aircraft manufacturing supply chain case study. The results demonstrate that publicly available textual data can provide an effective basis for reconstructing multi-tier supply chain networks, while the proposed reasoning mechanism improves the completeness of the reconstructed network. The resulting network representation enables the identification of upstream suppliers and structural dependencies that are difficult to capture through conventional data collection approaches. This study provides a scalable data-driven approach for representing complex supply chain systems and demonstrates how knowledge extraction and relational reasoning can be integrated to support supply chain mapping, structural analysis, and data-driven decision-making.
## This Repository
### Content
This repository provides with supplementary materials, including:
- Annotation guideline
- Annotations of entities and relations
- Annotations of triplets
- Annotation data samples
- Inter-annotators consistency check samples
- LLMs exact consistency and high consistency samples
- Kappa confusion matrices
- Codes for models, including attention-bilstm, bert-casrel
