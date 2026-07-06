import pickle

files = ['data/aligned_50.pkl', 'data/unaligned_50.pkl']

for fp in files:
    print(f"\n{'='*60}")
    print(f"FILE: {fp}")
    print('='*60)
    try:
        with open(fp, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"ERROR: {e}")
        continue

    print('top keys:', list(d.keys()))

    for split in ['train', 'valid', 'test']:
        if split not in d:
            print(f"  [{split}] MISSING")
            continue
        split_data = d[split]
        print(f"\n  [{split}]")
        print(f"    type: {type(split_data).__name__}")
        print(f"    keys: {list(split_data.keys())}")

        # Every key is a field name, value is array of (N_samples, ...)
        # Infer N_samples from any array field
        N = None
        for k, v in split_data.items():
            if hasattr(v, 'shape') and len(v.shape) >= 1:
                N = v.shape[0]
                break
        print(f"    N_samples: {N}")

        for k, v in split_data.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"    {k}: len={len(v)}")
            elif isinstance(v, str):
                print(f"    {k}: str len={len(v)}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:80]}")
