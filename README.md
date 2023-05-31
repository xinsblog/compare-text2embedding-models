* [对比各种文本转向量模型在q-d匹配任务上的效果](https://zhuanlan.zhihu.com/p/632888859)
* 效果对比如下

| model                                     | metric（spearman系数）     |
|-------------------------------------------|-------------|
| [junnyu/roformer_chinese_sim_char_small](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)    | 0.359512908 |
| [junnyu/roformer_chinese_sim_char_ft_small](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small) | 0.36817587  |
| [junnyu/roformer_chinese_sim_char_base](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)     | 0.463468448 |
| [junnyu/roformer_chinese_sim_char_ft_base](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)  | 0.511114737 |
| [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)    | 0.498120294 |
| [GanymedeNil/text2vec-base-chinese](https://huggingface.co/GanymedeNil/text2vec-base-chinese)         | 0.554429545 |
| [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese)        | 0.550098064 |
| [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model)                    | 0.641059161 |
