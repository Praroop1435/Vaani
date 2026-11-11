# Spot-check: 5 chunks (chunk size 300)
Open each PDF in `data/papers/` at the page shown and check: right section, right page, text matches, nothing important missing.

## 1706.03762v7#c013

- **Paper:** 1706.03762v7
- **Section:** 3.2.2 Model Architecture > Attention > Multi-Head Attention
- **PDF page:** 4-5
- **Tokens:** 300

> Instead of performing a single attention function with dmodel -dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk , dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv -dimensional To illustrate why the dot products get large, assume that the components of q and k are independent random P k variables with mean 0 and variance 1. Then their dot product, q · k = di=1 qi ki , has mean 0 and variance dk . output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. MultiHead(Q, K, V ) = Concat(head1 , ..., headh )W O where headi = Attention(QWiQ , KWiK , V WiV ) Where the projections are parameter matrices WiQ ∈ Rdmodel ×dk , WiK ∈ Rdmodel ×dk , WiV ∈ Rdmodel ×dv and W O ∈ Rhdv ×dmodel . In this work we employ h = 8 parallel attention layers, or heads.

## 1706.03762v7#c032

- **Paper:** 1706.03762v7
- **Section:** 6.2 Results > Model Variations
- **PDF page:** 8-9
- **Tokens:** 118

> To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the 5 We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013.

## 2410.00037v1#c043

- **Paper:** 2410.00037v1
- **Section:** 3.4.4 Model > Generative audio modeling > Inner Monologue
- **PDF page:** 15-16
- **Tokens:** 288

> While operating purely in the audio domain already yields convincing results (see Table 7), we observe that having Moshi also model the textual representation of its own speech is providing a scaffolding that increases the linguistic quality of its generation. Formally, we define a text stream W ∈ {1, . . . , NW }T derived from the sequence of text tokens obtained by applying the SentencePiece tokenizer (Kudo and Richardson, 2018) to the transcription of the audio corresponding to Moshi with Whisper (Radford et al., 2023), as described after. We insert W as the first sub-sequence in V , such that it acts as a prefix to the generation of semantic tokens. This can be seen as an extension of the hierarchical semanticto-acoustic generation introduced by Borsos et al. (2022). Note that we do not use the textual representation corresponding to the stream of the user, as transcribing this flux in real time would be challenging, and relying on an external ASR system contradicts our Moshi: a speech-text foundation model for real-time dialogue end-to-end speech-to-speech approach. Ablation studies in Section 5.3 show that among the design choices made for Moshi, Inner Monologue has one of the most critical impacts on the quality of generated speech. Aligning text and audio tokens.

## 2410.00037v1#c128

- **Paper:** 2410.00037v1
- **Section:** A Additional ablation on Mimi codec
- **PDF page:** 53
- **Tokens:** 302

> Table 17: Ablation study on hyperparameters of the Mimi codec. We evaluate semantic modeling by reporting the error rate on a phonetic ABX discriminability task. To evaluate reconstruction quality, we compute VisQOL and MOSNet. “Quantization rate” refers to applying quantization to the latent space only 50% of the time during training (independently from quantizer dropout), as described in Section 3.3. Quantization Rate Transformer in encoder Transformer in decoder WavLM distillation Split quantizer ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ABX (↓) VisQOL (↑) MOSNet (↑) 31.3% 31.4% 27.5% 29.0% 29.1% 27.4% 23.6% 23.3% 6.5% 10.8% 8.1% 8.0% 8.1% 2.37 2.30 2.30 2.25 2.65 2.69 2.72 2.82 2.13 2.68 2.49 2.36 2.72 2.85 2.82 2.93 2.94 2.86 2.83 2.89 2.89 2.87 2.84 2.71 2.88 2.89

## 2507.07955v2#c029

- **Paper:** 2507.07955v2
- **Section:** 2.2.3 H-Net Architecture > Dynamic Chunking (DC) > Ratio Loss
- **PDF page:** 8
- **Tokens:** 277

> Without explicit regularization, the model may converge to trivial solutions: either retaining nearly all vectors (negating computational benefits) or compressing excessively (losing critical information). Inspired by load balancing mechanisms in Mixture-of-Experts (MoE) models (Fedus, Zoph, and Shazeer 2022), which face similar challenges in maintaining balanced expert utilization, we introduce a ratio loss to guide compression: 𝐿 Lratio = 𝑁 ((𝑁 − 1)𝐹𝐺 + (1 − 𝐹 ) (1 − 𝐺)) , 𝑁 −1 𝐹= 1 ∑︁ 𝑏𝑡 , 𝐿 𝑡 =1 𝐿 𝐺= 1 ∑︁ 𝑝𝑡 , 𝐿 𝑡 =1 (10) where 𝐹 represents the fraction of vectors actually selected, 𝐺 denotes the average boundary probability, and 𝑁 controls the target compression ratio. Mechanistically, although 𝐹 is not differentiable, the network can be trained toward targeted compression ratios through 𝐺, which provides continuous feedback. When 𝐹 = 𝐺, the loss attains a minimum of Lratio = 1 when 𝐹 = 𝐺 = 𝑁1 . Interestingly, the loss can theoretically fall below 1 when 𝐹 ≠ 𝐺 (e.g., 𝐹 = 𝑁1 + 𝜖 and 𝐺 = 𝑁1 − 𝜖), which we indeed observe during training. Despite this theoretical possibility, the loss effectively guides the model toward the desired compression ratio in practice.
