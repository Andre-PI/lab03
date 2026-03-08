import numpy as np

np.random.seed(42)

d_model = 64
d_k = 8
vocab_size = 10000


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


print("=" * 50)
print("TAREFA 1")
print("=" * 50)

seq_len = 5
mask = create_causal_mask(seq_len)
print("\nMascara gerada (seq_len=5):")
print(mask)

Q_test = np.random.randn(seq_len, d_k)
K_test = np.random.randn(seq_len, d_k)

scores = Q_test @ K_test.T / np.sqrt(d_k)
scores_masked = scores + mask
attn_weights = softmax(scores_masked)

print("\nPesos de atencao apos mascara:")
print(np.round(attn_weights, 4))

print("\n" + "=" * 50)
print("TAREFA 2")
print("=" * 50)

encoder_output = np.random.randn(1, 10, d_model)
decoder_state  = np.random.randn(1,  4, d_model)

Wq = np.random.randn(d_model, d_model) * 0.1
Wk = np.random.randn(d_model, d_model) * 0.1
Wv = np.random.randn(d_model, d_model) * 0.1


def cross_attention(encoder_out, decoder_state):
    Q = decoder_state @ Wq
    K = encoder_out   @ Wk
    V = encoder_out   @ Wv

    scores  = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
    weights = softmax(scores)
    output  = weights @ V
    return output, weights


cross_out, cross_weights = cross_attention(encoder_output, decoder_state)
print(f"\nShape encoder_output : {encoder_output.shape}")
print(f"Shape decoder_state  : {decoder_state.shape}")
print(f"Shape saida cross-att: {cross_out.shape}")
print(f"\nPesos cross-attention (4 tokens do decoder x 10 tokens do encoder):")
print(np.round(cross_weights[0], 3))

print("\n" + "=" * 50)
print("TAREFA 3")
print("=" * 50)

id2word = {i: f"palavra_{i}" for i in range(vocab_size)}
id2word[0] = "<START>"
id2word[1] = "<EOS>"
id2word[2] = "O"
id2word[3] = "rato"
id2word[4] = "roeu"
id2word[5] = "a"
id2word[6] = "roupa"
id2word[7] = "do"
id2word[8] = "rei"

W_out = np.random.randn(d_model, vocab_size) * 0.01


def generate_next_token(current_sequence, encoder_out):
    seq_len = len(current_sequence)
    decoder_input = np.random.randn(1, seq_len, d_model)

    mask = create_causal_mask(seq_len)
    scores  = decoder_input @ decoder_input.transpose(0, 2, 1) / np.sqrt(d_model)
    scores += mask
    weights = softmax(scores)
    self_att_out = weights @ decoder_input

    x = layer_norm(decoder_input + self_att_out)

    cross_out, _ = cross_attention(encoder_out, x)
    x = layer_norm(x + cross_out)

    last_vec = x[:, -1, :]
    logits   = last_vec @ W_out
    probs    = softmax(logits)
    return probs[0]


sequence = ["<START>"]
max_tokens = 10

print(f"\nSequencia inicial: {sequence}")

step = 0
while step < max_tokens:
    probs     = generate_next_token(sequence, encoder_output)
    next_id   = int(np.argmax(probs))
    next_word = id2word[next_id]

    sequence.append(next_word)
    print(f"Passo {step+1}: gerou '{next_word}' (id={next_id}, prob={probs[next_id]:.4f})")

    if next_word == "<EOS>":
        print("\nToken <EOS> encontrado. Geracao encerrada.")
        break

    step += 1

print(f"\nFrase final gerada: {' '.join(sequence)}")
