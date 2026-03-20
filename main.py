from transformer_lens import HookedTransformer

model: HookedTransformer = HookedTransformer.from_pretrained("pythia-14M")

logits, cache = model.run_with_cache("The sky is bluue and the grass is green.")

print(logits)

print(cache.keys)
