<div align="center">

# EPO: Hierarchical LLM Agents with Environment Preference Optimization

Qi Zhao*, Haotian Fu*, Chen Sun, George Konidaris

EMNLP 2024

![](assets/main.gif)
</div>

Long-horizon decision-making tasks present significant challenges for LLM-based agents due to the need for extensive planning over multiple steps. In this paper, we propose a hierarchical framework that decomposes complex tasks into manageable subgoals, utilizing separate LLMs for subgoal prediction and low-level action generation. To address the challenge of creating training signals for unannotated datasets, we develop a reward model that leverages multimodal environment feedback to automatically generate reward signals. We introduce Environment Preference Optimization (EPO), a novel method that generates preference signals from the environment's feedback and uses them to train LLM-based agents. Extensive experiments on ALFRED demonstrate the state-of-the-art performance of our framework, achieving first place on the ALFRED public leaderboard and showcasing its potential to improve long-horizon decision-making in diverse environments.


## Contents
- [Our Paper](#Our-Paper)
- [License](#License)

## Our Paper 

Our paper is available on [Arxiv](https://arxiv.org/abs/2408.16090). If you find our work  useful, please consider citing us. 
```bibtex
@article{zhao2024epo,
  title   = {EPO: Hierarchical LLM Agents with Environment Preference Optimization},
  author  = {Qi Zhao and Haotian Fu and Chen Sun and George Konidaris},
  journal = {EMNLP},
  year    = {2024}
}
```