# Table check: 3 tables from the LaTeX source

Compare each with the PDF page shown. Check: right table number and page, header names, and a few cell values.

## 1706.03762v7 Table 2 (section 5.4, 1 chunk(s), 389 tokens)

Table 2 (page 8): The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

| Model | BLEU EN-DE | BLEU EN-FR | Training Cost (FLOPs) EN-DE | Training Cost (FLOPs) EN-FR |
| --- | --- | --- | --- | --- |
| ByteNet | 23.75 |  |  |  |
| Deep-Att + PosUnk |  | 39.2 |  | 1.0·10^20 |
| GNMT + RL | 24.6 | 39.92 | 2.3·10^19 | 1.4·10^20 |
| ConvS2S | 25.16 | 40.46 | 9.6·10^18 | 1.5·10^20 |
| MoE | 26.03 | 40.56 | 2.0·10^19 | 1.2·10^20 |
| Deep-Att + PosUnk Ensemble |  | 40.4 |  | 8.0·10^20 |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8·10^20 | 1.1·10^21 |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7·10^19 | 1.2·10^21 |
| Transformer (base model) | 27.3 | 38.1 | 3.3·10^18 | 3.3·10^18 |
| Transformer (big) | 28.4 | 41.8 | 2.3·10^19 | 2.3·10^19 |


## 1706.03762v7 Table 3 (section 6.2, 2 chunk(s), 940 tokens)

Table 3 (page 9): Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

|  | N | d_model | d_ff | h | d_k | d_v | P_drop | ε_ls | train steps | PPL (dev) | BLEU (dev) | params ×10^6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) |  |  |  | 1 | 512 | 512 |  |  |  | 5.29 | 24.9 |  |
|  |  |  |  | 4 | 128 | 128 |  |  |  | 5.00 | 25.5 |  |
|  |  |  |  | 16 | 32 | 32 |  |  |  | 4.91 | 25.8 |  |
|  |  |  |  | 32 | 16 | 16 |  |  |  | 5.01 | 25.4 |  |
| (B) |  |  |  |  | 16 |  |  |  |  | 5.16 | 25.1 | 58 |
|  |  |  |  |  | 32 |  |  |  |  | 5.01 | 25.4 | 60 |
| (C) | 2 |  |  |  |  |  |  |  |  | 6.11 | 23.7 | 36 |
|  | 4 |  |  |  |  |  |  |  |  | 5.19 | 25.3 | 50 |
|  | 8 |  |  |  |  |  |  |  |  | 4.88 | 25.5 | 80 |

|  | N | d_model | d_ff | h | d_k | d_v | P_drop | ε_ls | train steps | PPL (dev) | BLEU (dev) | params ×10^6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 256 |  |  | 32 | 32 |  |  |  | 5.75 | 24.5 | 28 |
|  |  | 1024 |  |  | 128 | 128 |  |  |  | 4.66 | 26.0 | 168 |
|  |  |  | 1024 |  |  |  |  |  |  | 5.12 | 25.4 | 53 |
|  |  |  | 4096 |  |  |  |  |  |  | 4.75 | 26.2 | 90 |
| (D) |  |  |  |  |  |  | 0.0 |  |  | 5.77 | 24.6 |  |
|  |  |  |  |  |  |  | 0.2 |  |  | 4.95 | 25.5 |  |
|  |  |  |  |  |  |  |  | 0.0 |  | 4.67 | 25.3 |  |
|  |  |  |  |  |  |  |  | 0.2 |  | 5.47 | 25.7 |  |
| (E) |  | positional embedding instead of sinusoids | positional embedding instead of sinusoids | positional embedding instead of sinusoids | positional embedding instead of sinusoids | positional embedding instead of sinusoids | positional embedding instead of sinusoids | positional embedding instead of sinusoids |  | 4.92 | 25.7 |  |
| big | 6 | 1024 | 4096 | 16 |  |  | 0.3 |  | 300K | 4.33 | 26.4 | 213 |


## 2507.07955v2 Table 4 (section 3.2, 1 chunk(s), 354 tokens)

Table 4 (page 17): Architecture details and model benchmarks for Chinese and code models. BPIC (defined in ) denotes the compression between the main network and outermost stage (bytes). Each H-Net used (3,3)-DC, targeting an inner downsampling ratio of 9. However, the resulting BPIC was significantly different, indicating that code is much easier to compress than Chinese. In terms of results, H-Net (2-stage) performs better than both H-Net (space) and BPE Transformer on Chinese, which is reflected in the downstreams. On the other hand, H-Net (2-stage) achieves similar performance to H-Net (space) on code, and both H-Net models perform significantly better than Transformer.

| Model | Chinese BPIC | Chinese Main arch. | Chinese Val. BPB ↓ | Chinese XW-zh. acc. ↑ | Code BPIC | Code Main arch. | Code Val. BPB ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | 3.62 | T15 | 0.7404 | 0.599 | 3.58 | T13 | 0.3376 |
| H-Net (space) | 3.38 | T19 | 0.7478 | 0.629 | 7.97 | T40 | 0.3163 |
| H-Net (2-stage) | 5.81 | T30 | 0.7032 | 0.663 | 7.23 | T28 | 0.3161 |

