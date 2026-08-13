# Modern Decoder-Only LLMs: Handwritten Theory Notes

## 1. Same Job, Updated Machinery

A modern decoder still does the same job as the earlier teaching model: read the visible token prefix and predict what comes next. The upgrade is inside each block, not in the learning objective.

Think of the residual stream as a **shared notebook passed through the model**. Every block reads it, writes a useful correction, and passes the updated notebook onward. Modern components make those reads and corrections cheaper, steadier, or more selective.

The block now looks like this:

```text
token IDs
-> shared token embeddings
-> RMSNorm
-> grouped-query attention with RoPE
-> residual addition
-> RMSNorm
-> SwiGLU
-> residual addition
-> final RMSNorm
-> vocabulary scores through the shared embedding table
```

The important continuity is easy to miss: **modern LLMs are refined Transformers, not a different species of model.**

## 2. RMSNorm Is a Volume Control

LayerNorm performs two jobs: it recenters a token vector and rescales it. RMSNorm keeps only the scale-control job.

Mental model:

- **LayerNorm:** move the signal back to center, then set its volume.
- **RMSNorm:** leave its offset alone and set only its volume.

RMSNorm operates independently on every token. It does not mix tokens, add context, or decide which features matter. It simply keeps feature magnitudes in a manageable range before attention and the feed-forward network use them.

The durable idea is: **normalization controls the condition of the residual stream; it does not create the model's knowledge.**

## 3. SwiGLU Is a Learned Feature Gate

The feed-forward network is each token's private workspace. Attention gathers information from other positions; the FFN decides how to transform the gathered features.

A basic GELU FFN has one expanded feature path. SwiGLU creates two:

- **Candidate branch:** what transformed content could be useful?
- **Gate branch:** how much of each candidate feature should pass?

The branches meet feature by feature, then project back to model width. This gives the token a learned filter rather than sending every expanded feature through one activation path.

Memory aid: **attention chooses where to read; SwiGLU chooses what transformed features to keep.**

SwiGLU is not attention and it is not a mixture-of-experts router. It acts inside one token position using the same weights at every position.

## 4. GQA Shares Keys and Values, Not Questions

Attention heads can ask different questions about the same sequence. The expensive part during generation is storing the past keys and values that those questions search.

Three ownership patterns form a spectrum:

| Pattern | Query heads | Key/value ownership | Mental model |
|---|---:|---|---|
| MHA | Many | One K/V pair per query head | Every researcher keeps a private index |
| GQA | Many | Small groups share K/V pairs | Research teams share an index |
| MQA | Many | All queries share one K/V pair | Everyone uses one common index |

Grouped-query attention keeps distinct query projections while sharing key/value projections within groups. The model can still ask several questions, but it stores fewer versions of the searchable history.

The trade-off is simple: **more sharing reduces K/V parameters and cache memory, but also reduces how independently heads can represent searchable history.**

The model width must divide cleanly into query heads, and query heads must divide cleanly into K/V groups. Invalid groupings are architecture errors, not tuning choices.

## 5. RoPE Puts Position Inside the Comparison

Attention needs order. Without position, the same words in a different arrangement look like the same set of token vectors.

RoPE rotates query and key feature pairs according to token position. When a rotated query meets a rotated key, their comparison reveals the distance between their positions.

Three details matter:

- Apply RoPE after queries and keys have been split into heads.
- Rotate queries and keys because they decide where to look.
- Do not rotate values because values carry the content being retrieved.

Shifting two tokens together preserves their relative gap, so their positional relationship stays comparable. RoPE changes orientation, not vector length.

Memory aid: **embeddings say what a token is; RoPE helps attention notice where tokens sit relative to one another.**

## 6. The Complete Modern Block

A modern pre-normalized block performs two corrections to the residual stream:

1. RMSNorm prepares the current token states; GQA lets each token retrieve permitted history; the result is added back.
2. RMSNorm prepares that updated state; SwiGLU performs private feature selection; the result is added back.

Sequence length and model width stay unchanged through the block. What changes is the meaning carried at each position.

The final model stacks these blocks, normalizes once more, and maps each position to vocabulary scores. If the output head is tied to the token embedding table, the same learned token matrix serves two directions:

- token ID -> vector when reading;
- vector -> token scores when predicting.

Weight tying saves a second large vocabulary matrix and keeps input/output token geometry connected. Equal values are not enough to prove tying; both paths must share the same stored parameter.

## 7. Practical Failure Modes

- **Treating modern components as a new objective:** the model still learns by next-token prediction.
- **Claiming RMSNorm is always better:** it has a simpler scale-only contract, not a universal quality guarantee.
- **Calling SwiGLU attention:** it transforms one token's features; it does not route across positions.
- **Confusing GQA with fewer query heads:** queries remain numerous; keys and values are shared.
- **Using an invalid head grouping:** query heads must divide evenly across K/V groups.
- **Applying RoPE to values:** values carry retrieved content and should not receive positional rotation.
- **Applying RoPE before head splitting:** feature pairs may cross the wrong head boundary.
- **Counting a tied output head twice:** tied storage is one parameter matrix with two uses.
- **Reading attention maps as explanations:** routing weights are only one contribution to the final prediction.

The durable model is: **RMSNorm steadies the stream, GQA retrieves history economically, RoPE makes order visible, SwiGLU filters private features, and residual paths preserve continuity through depth.**
