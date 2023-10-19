Struct-XLM
---
The code of paper accepted in EMNLP2023, _Struct-XLM: A Structure Discovery Multilingual Language Model for Enhancing Cross-lingual Transfer through Reinforcement Learning_.

### Abstract

---
Cross-lingual transfer learning heavily relies on well-aligned cross-lingual representations. Syntactic structure is recognized as beneficial for cross-lingual transfer, but limited researches utilize it for aligning representation in multilingual pre-trained language models (PLMs). Additionally, existing methods require syntactic labels that are difficult to obtain and of poor quality for low-resource languages. To address this gap, we propose Struct-XLM, a novel multilingual language model that leverages reinforcement learning (RL) to autonomously discover universal syntactic structure for improved cross-lingual representation alignment. Struct-XLM integrates a policy network (PNet) and a translation ranking task. The PNet is designed to discover structural information, which is then integrated into the last layer of the PLM through the structural multi-head attention module to obtain structural representation. The translation ranking task obtains a delayed reward based on the structural representation to optimize the PNet while improving the alignment of cross-lingual representation. Through extensive experiments on the XTREME cross-lingual understanding benchmark, we validate the effectiveness of our approach, demonstrating significant improvements in cross-lingual transfer and enhanced representation alignment compared to the baseline PLM, while producing competitive results compared to state-of-the-art methods.

This resource contains two directories ```src``` and ```data```, the Struct-XLM and seven benchmark tasks evalution code in ```src```, and all the train and test datasets in ```data``` or ```src/xtreme_7/data```.


### Environment

---
- GPU       NVIDIA RTX A6000  48G
- python    3.7.13
- torch     1.12.1
- cuda      11.7
- transformers 4.30.0

### Usage

```
bash run.sh
```

=======
# Struct-XLM
The code of paper accepted in EMNLP2023, Struct-XLM: A Structure Discovery Multilingual Language Model for Enhancing Cross-lingual Transfer through Reinforcement Learning.
>>>>>>> d9c883d4da0e86511be7921c43c0f48525c8b4d6
