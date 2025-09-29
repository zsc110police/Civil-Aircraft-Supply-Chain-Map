# Civil-Aircraft-Supply-Chain-Knowledge-Graphs
The data and code are associated with the paper *'Constructing Supply Chain Knowledge Graphs by Automatically Extracting Organization–Product Relations from News Articles with a Joint Relation Extraction Model'* by XXX, China.
## Abstract
News texts, as a vital source for constructing supply chain knowledge graphs, often contain implicit relations that are expressed indirectly through products, which poses significant challenges for relation extraction. Focusing on the civil aircraft industry supply chain, this study proposes a relation extraction strategy that simultaneously captures both “organization–organization” and “organization–product” relations, and builds a corresponding supply chain knowledge graph. A Chinese corpus comprising 19,811 sentences was constructed through web crawling and manual collection, and 4,068 positive samples were annotated with the assistance of both human annotators and large language models (LLMs), yielding a total of 30,242 labels (17,993 entities and 12,249 relations). To address the common issue of overlapping entities, the Bert-casrel model was employed for joint entities and relations extraction. Experimental results demonstrate that Bert casrel outperforms traditional pipeline methods in extracting supply chain relations. Furthermore, visualization with Neo4j and Cypher queries enables effective exploration of both inter-organizational and organization–product relations, while comparison with enterprise websites and supply chain databases validates the accuracy and extensibility of the proposed knowledge graph. Overall, the proposed approach exhibits strong generalizability and provides a methodological reference for constructing supply chain knowledge graphs in other domains.
## This Repository
### Content
This repository provides with supplementary materials, including:
- Annotation guideline
- Annotations of entities and relations
- Annotations of triplets
- Annotation data 200 samples
- Inter-annotators consistency check samples
- LLMs exact consistency and high consistency samples
- Kappa confusion matrices
- Codes for models, including attention-bilstm, bert-casrel, bert-casrel with constrained triplets
