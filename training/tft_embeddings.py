"""forward hook으로 TFT의 output_layer 직전 32차원(=state_size) 은닉 표현을 뽑는다.
tft-torch 벤더 코드(tft-torch/tft_torch/tft.py)는 수정하지 않는다 -- gated_poswise_ff를 만드는
model.pos_wise_ff_gating 모듈의 forward 출력을 hook으로 가로챈다(tft.py:983-987 참고)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.signal_data import sample_meta
from training.train import _to_device_batch


def extract_embeddings(model, dataset, device, batch_size: int = 256) -> tuple[list[tuple[str, str]], np.ndarray]:
    keys = [(tk, d) for tk, d, _ in sample_meta(dataset)]
    model = model.to(device)
    model.eval()
    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.pos_wise_ff_gating.register_forward_hook(_hook)
    chunks = []
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = dict(batch)
                batch.pop("label", None)
                batch = _to_device_batch(batch, device)
                captured.clear()
                model(batch)
                assert len(captured) == 1, "expected exactly one pos_wise_ff_gating call per forward"
                emb = captured[0].squeeze(1)  # [batch, output_seq_len=1, state_size] -> [batch, state_size]
                chunks.append(emb.cpu().numpy())
    finally:
        handle.remove()
    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)
    return keys, embeddings
