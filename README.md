# Civil-Aircraft-Supply-Chain-Maps
The data and code are associated with the paper *'Constructing Supply Chain Knowledge Graphs by Automatically Extracting Organization–Product Relations from News Articles with a Joint Relation Extraction Model'* by XXX, China.
## Abstract
Extracting supply chain maps from news articles using deep learning has proven feasible and effective. However, numerous implicit relations are expressed indirectly through products, which significantly reduces the effectiveness of Relation Extraction (RE). To address this problem, this study proposes a RE strategy that simultaneously extracts “organization–organization” and “organization–product” relations. The civil aircraft supply chain is chosen as the study case, since it represents a highly complex system with low visibility. During data collection and processing, a hybrid annotation framework combining human annotators and large language models was employed to label news texts, yielding 30,242 labels. To mitigate the common problem of overlapping entities, the BERT-CasRel model, which comprises a pre-trained language model and a cascade binary tagging framework, was employed for joint relation extraction. Experimental results show that BERT-CasRel outperforms traditional pipeline methods when NER error is large, while the pipeline approach achieves comparable or superior results when NER error is small. Furthermore, visualization results demonstrate that the proposed method enables flexible inquiries into supply chain relationships, such as connections between organizations and products, suppliers and manufacturers of specific product models, products produced by special organization. This study contributes a practical and generalizable methodological framework for constructing more complete supply chain maps, offering data and technology support for scientific decision-making.
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
