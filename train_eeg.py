"""
Overfit test: per-channel EnCodec on EEG data.

Each EEG epoch has shape [C, T] where C varies (30-59) and T varies (64-7680).
Strategy (Option A): reshape [C, T] -> [C, 1, T], run 1-channel EnCodec on each
electrode independently, then reshape back. No model code changes needed.

Model: 1-channel, 256 Hz, ratios=[2,2,2,2] (16x stride -> 16 fps), 8 codebooks.
Loss: MSE reconstruction + VQ commitment loss. No discriminator.
"""

import math
import os
import argparse
import contextlib
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from encodec.model import EncodecModel
from encodec.modules.seanet import SEANetEncoder, SEANetDecoder
from encodec.msstftd import MultiScaleSTFTDiscriminator
from encodec.quantization.vq import ResidualVectorQuantizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = '/data/datasets/bci/pt3d_eval/pt3d_varying_length'
SAMPLE_RATE = 256
RATIOS = [2, 2, 2, 2]           # 16x total stride
TOTAL_STRIDE = 16               # product of RATIOS
FRAME_RATE = math.ceil(SAMPLE_RATE / TOTAL_STRIDE)  # 16 fps
N_Q = 8                         # residual codebooks
BANDWIDTH = N_Q * math.log2(1024) * FRAME_RATE / 1000  # 1.28 kbps

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Discriminator loss weights
LAMBDA_ADV = 0.1   # generator adversarial loss (small — FM carries the signal)
LAMBDA_FM  = 10.0  # feature matching loss (workhorse for perceptual quality)

# EEG-appropriate STFT windows at 256 Hz (audio uses [1024,2048,512])
# 128 samples = 0.5 s, 64 = 0.25 s, 32 = 0.125 s; hop = window/4
DISC_N_FFTS      = [128, 64, 32]
DISC_HOP_LENGTHS = [32,  16,  8]
DISC_WIN_LENGTHS = [128, 64, 32]
DISC_MIN_T       = max(DISC_N_FFTS)  # skip disc for sequences shorter than this


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_eeg_model(kmeans_init: bool = True, kmeans_iters: int = 50) -> EncodecModel:
    """1-channel causal EnCodec tuned for 256 Hz EEG."""
    encoder = SEANetEncoder(
        channels=1,
        dimension=128,
        n_filters=32,
        ratios=RATIOS,
        lstm=1,
        causal=True,
    )
    decoder = SEANetDecoder(
        channels=1,
        dimension=128,
        n_filters=32,
        ratios=RATIOS,
        lstm=1,
        causal=True,
    )
    quantizer = ResidualVectorQuantizer(
        dimension=128,
        n_q=N_Q,
        bins=1024,
        kmeans_init=kmeans_init,
        kmeans_iters=kmeans_iters,
    )
    model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        target_bandwidths=[BANDWIDTH],
        sample_rate=SAMPLE_RATE,
        channels=1,
        normalize=False,
        segment=None,
    )
    model.set_target_bandwidth(BANDWIDTH)
    return model


def make_eeg_discriminator() -> MultiScaleSTFTDiscriminator:
    return MultiScaleSTFTDiscriminator(
        filters=32,
        in_channels=1,
        n_ffts=DISC_N_FFTS,
        hop_lengths=DISC_HOP_LENGTHS,
        win_lengths=DISC_WIN_LENGTHS,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGEpochDataset(Dataset):
    """Each item is one EEG epoch: float32 tensor [C, T], z-scored per channel."""

    def __init__(self, data_dir: str, max_files: int | None = None):
        pt_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.pt'))
        if max_files is not None:
            pt_files = pt_files[:max_files]

        self.epochs: list[torch.Tensor] = []
        for fname in pt_files:
            chunk = torch.load(
                os.path.join(data_dir, fname),
                map_location='cpu',
                weights_only=False,
            )
            for ep in chunk['data']:
                self.epochs.append(ep.float())  # [C, T]

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.epochs[idx]  # [C, T]
        raw_length = x.shape[-1]

        # per-channel z-score so every electrode is on the same scale. Do this
        # before padding so synthetic tail samples cannot shift the real signal.
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        x = (x - mu) / sigma

        # pad T to a multiple of the encoder's total stride
        T = x.shape[-1]
        rem = T % TOTAL_STRIDE
        if rem:
            x = F.pad(x, (0, TOTAL_STRIDE - rem))

        return x, raw_length  # [C, T_padded], raw T


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

N_TRACE_CHANNELS = 6   # channels to show in time-trace plot
VIS_DIR = 'vis'


def resolve_data_dir(data_dir: str) -> str:
    if os.path.isdir(data_dir):
        return data_dir
    local_data_dir = os.path.expanduser('~/pt3d_varying_length')
    if data_dir == DATA_DIR and os.path.isdir(local_data_dir):
        return local_data_dir
    return data_dir


def visualize(model: EncodecModel, x: torch.Tensor, epoch: int, raw_length: int | None = None) -> None:
    """
    x: [C, T] on device (already z-scored).
    Saves two figures per call:
      vis/epoch_{n}_traces.png  — original vs reconstructed time traces
      vis/epoch_{n}_psd.png     — mean PSD original vs reconstructed
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    model.eval()
    with torch.no_grad():
        recon, loss_recon, _, _codes = forward_eeg(model, x, raw_length=raw_length)
    model.train()

    if raw_length is not None:
        x = x[..., :raw_length]
        recon = recon[..., :raw_length]

    orig = x.cpu().numpy()       # [C, T_raw]
    rec  = recon.cpu().numpy()   # [C, T_raw]
    C, T = orig.shape
    t = np.arange(T) / SAMPLE_RATE

    # --- time traces ---
    n_ch = min(N_TRACE_CHANNELS, C)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, orig[i], color='steelblue', lw=0.8, label='original')
        ax.plot(t, rec[i],  color='tomato',    lw=0.8, label='recon', alpha=0.85)
        r = np.corrcoef(orig[i], rec[i])[0, 1]
        ax.set_ylabel(f'ch {i}\nr={r:.3f}', fontsize=8)
        ax.set_yticks([])
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'epoch {epoch}  |  recon MSE={loss_recon.item():.4f}  |  {C} channels', fontsize=10)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, f'epoch_{epoch:04d}_traces.png'), dpi=120)
    plt.close(fig)

    # --- power spectral density ---
    freqs = np.fft.rfftfreq(T, d=1.0 / SAMPLE_RATE)
    psd_orig = (np.abs(np.fft.rfft(orig, axis=-1)) ** 2).mean(axis=0)
    psd_rec  = (np.abs(np.fft.rfft(rec,  axis=-1)) ** 2).mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(freqs, psd_orig, color='steelblue', lw=1.2, label='original')
    ax.semilogy(freqs, psd_rec,  color='tomato',    lw=1.2, label='recon', alpha=0.85)
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power (log scale)')
    ax.set_title(f'epoch {epoch}  |  mean PSD across {C} channels')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, f'epoch_{epoch:04d}_psd.png'), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def forward_eeg(model: EncodecModel, x: torch.Tensor, raw_length: int | None = None):
    """
    x: [C, T] — one epoch on device.
    Returns (recon [C, T], loss_recon, loss_commit, codes [n_q, C, T_frames]).
    """
    C, T = x.shape
    x_flat = x.unsqueeze(1)                                    # [C, 1, T]

    z = model.encoder(x_flat)                                  # [C, 128, T//16]
    qres = model.quantizer(z, model.frame_rate, model.bandwidth)
    recon = model.decoder(qres.quantized)                      # [C, 1, T]

    if raw_length is None:
        loss_recon = F.mse_loss(recon, x_flat)
    else:
        loss_recon = F.mse_loss(recon[..., :raw_length], x_flat[..., :raw_length])
    loss_commit = qres.penalty
    return recon.squeeze(1), loss_recon, loss_commit, qres.codes


# ---------------------------------------------------------------------------
# Dummy architecture inspection
# ---------------------------------------------------------------------------

@dataclass
class LayerTrace:
    name: str
    module: str
    input_shape: tuple[int, ...] | str
    output_shape: tuple[int, ...] | str


def _shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value]
    if hasattr(value, 'quantized') and hasattr(value, 'codes'):
        return {
            'quantized': tuple(value.quantized.shape),
            'codes': tuple(value.codes.shape),
            'bandwidth': tuple(value.bandwidth.shape),
            'penalty': tuple(value.penalty.shape) if isinstance(value.penalty, torch.Tensor) else value.penalty,
        }
    return type(value).__name__


@contextlib.contextmanager
def trace_eeg_layers(model: EncodecModel):
    """Collect layer-level input/output shapes for the EEG forward path."""
    traces: list[LayerTrace] = []
    handles = []

    def add_hook(name: str):
        def hook(module, inputs, output):
            input_value = inputs[0] if len(inputs) == 1 else inputs
            traces.append(
                LayerTrace(
                    name=name,
                    module=module.__class__.__name__,
                    input_shape=_shape(input_value),
                    output_shape=_shape(output),
                )
            )
        return hook

    for prefix, seq in (('encoder', model.encoder.model), ('decoder', model.decoder.model)):
        for idx, module in enumerate(seq):
            handles.append(module.register_forward_hook(add_hook(f'{prefix}.{idx:02d}')))
    for idx, module in enumerate(model.quantizer.vq.layers):
        handles.append(module.register_forward_hook(add_hook(f'quantizer.{idx:02d}')))
    handles.append(model.quantizer.register_forward_hook(add_hook('quantizer.rvq')))

    try:
        yield traces
    finally:
        for handle in handles:
            handle.remove()


def make_dummy_eeg_epoch(channels: int = 32, length: int = 768, seed: int = 0) -> torch.Tensor:
    """
    Create one z-scored EEG-like epoch with shape [C, T].

    The signal mixes low-frequency drift, theta/alpha/beta components, channel-specific
    phase/amplitude, and noise. Length is padded to the model stride just like the dataset.
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32) / SAMPLE_RATE
    base_freqs = torch.tensor([1.5, 6.0, 10.0, 22.0], dtype=torch.float32)
    epoch = []

    for channel in range(channels):
        amps = torch.rand(len(base_freqs), generator=generator) * torch.tensor([0.4, 0.7, 1.0, 0.3])
        phases = torch.rand(len(base_freqs), generator=generator) * (2 * math.pi)
        signal = sum(
            amps[i] * torch.sin(2 * math.pi * base_freqs[i] * t + phases[i])
            for i in range(len(base_freqs))
        )
        slow_drift = 0.15 * torch.sin(2 * math.pi * (0.15 + 0.02 * channel) * t)
        noise = 0.08 * torch.randn(length, generator=generator)
        epoch.append(signal + slow_drift + noise)

    x = torch.stack(epoch, dim=0)
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
    x = (x - mu) / sigma
    rem = x.shape[-1] % TOTAL_STRIDE
    if rem:
        x = F.pad(x, (0, TOTAL_STRIDE - rem))
    return x


def _format_shape(value) -> str:
    if isinstance(value, tuple):
        return '[' + ', '.join(str(v) for v in value) + ']'
    if isinstance(value, list):
        return '[' + ', '.join(_format_shape(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{key}: {_format_shape(val)}' for key, val in value.items()) + '}'
    return str(value)


def run_dummy_arch_check(channels: int, length: int, seed: int) -> None:
    device = torch.device(DEVICE)
    model = make_eeg_model(kmeans_init=False).to(device)
    model.eval()
    x = make_dummy_eeg_epoch(channels=channels, length=length, seed=seed).to(device)

    with torch.no_grad(), trace_eeg_layers(model) as traces:
        recon, loss_recon, loss_commit, codes = forward_eeg(model, x)

    print('dummy EEG architecture check')
    print(f'input epoch:              {list(x.shape)}  [channels, time]')
    print(f'per-channel model input:  {[x.shape[0], 1, x.shape[1]]}  [batch=C, 1, time]')
    print(f'sample_rate={SAMPLE_RATE} Hz | total_stride={TOTAL_STRIDE} | frame_rate={model.frame_rate} fps')
    print(f'bandwidth={model.bandwidth:.4f} kbps | n_q={N_Q} | bins={model.quantizer.bins}')
    print('')
    print(f'{"stage":<18} {"module":<24} {"input":<24} output')
    print('-' * 96)
    for trace in traces:
        print(
            f'{trace.name:<18} '
            f'{trace.module:<24} '
            f'{_format_shape(trace.input_shape):<24} '
            f'{_format_shape(trace.output_shape)}'
        )
    print('-' * 96)
    print(f'codes:                    {list(codes.shape)}  [n_q, channels, frames]')
    print(f'reconstruction:           {list(recon.shape)}  [channels, time]')
    print(f'reconstruction MSE:       {loss_recon.item():.6f}')
    print(f'commitment loss:          {loss_commit.item():.6f}')


def train(
    model: EncodecModel,
    dataset: EEGEpochDataset,
    disc: MultiScaleSTFTDiscriminator | None = None,
    n_epochs: int = 500,
    lr: float = 1e-3,
    disc_warmup: int = 50,   # epochs before discriminator is enabled
    log_every: int = 10,
    vis_every: int = 50,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: b[0],
    )
    optimizer_g = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(disc.parameters(), lr=lr * 3, betas=(0.5, 0.9)) if disc else None

    probe, probe_raw_length = dataset[0]
    probe = probe.to(DEVICE)

    model.train()
    if disc is not None:
        disc.train()

    for epoch in range(1, n_epochs + 1):
        use_disc = disc is not None and epoch > disc_warmup
        total_recon = total_commit = total_adv = total_fm = total_d = 0.0
        code_counts = torch.zeros(N_Q, 1024)

        for x, raw_length in loader:
            x = x.to(DEVICE)
            recon, loss_recon, loss_commit, codes = forward_eeg(model, x, raw_length=raw_length)

            x_real = x[..., :raw_length]
            recon_real = recon[..., :raw_length]
            x_d = x_real.unsqueeze(1)      # [C, 1, T_raw]
            r_d = recon_real.unsqueeze(1)  # [C, 1, T_raw]

            # --- generator step ---
            if use_disc and raw_length >= DISC_MIN_T:
                logits_real, fmaps_real = disc(x_d.detach())
                logits_fake, fmaps_fake = disc(r_d)

                loss_adv = sum(-lf.mean() for lf in logits_fake) / len(logits_fake)
                loss_fm = sum(
                    F.l1_loss(ff, fr.detach())
                    for fmr, fmf in zip(fmaps_real, fmaps_fake)
                    for fr, ff in zip(fmr, fmf)
                ) / sum(len(fmr) for fmr in fmaps_real)
                g_loss = loss_recon + loss_commit + LAMBDA_ADV * loss_adv + LAMBDA_FM * loss_fm
                total_adv += loss_adv.item()
                total_fm  += loss_fm.item()
            else:
                g_loss = loss_recon + loss_commit

            optimizer_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_g.step()

            # --- discriminator step ---
            if use_disc and raw_length >= DISC_MIN_T:
                logits_real, _ = disc(x_d.detach())
                logits_fake, _ = disc(r_d.detach())

                loss_d = sum(
                    F.relu(1 - lr).mean() + F.relu(1 + lf).mean()
                    for lr, lf in zip(logits_real, logits_fake)
                ) / len(logits_real)

                optimizer_d.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                optimizer_d.step()
                total_d += loss_d.item()

            total_recon  += loss_recon.item()
            total_commit += loss_commit.item()

            codes_cpu = codes.detach().cpu()
            for q in range(N_Q):
                flat = codes_cpu[q].flatten().long()
                code_counts[q].scatter_add_(0, flat, torch.ones(flat.shape[0]))

        if epoch % log_every == 0:
            n = len(loader)
            util = (code_counts > 0).float().mean(dim=-1)
            p    = code_counts / code_counts.sum(dim=-1, keepdim=True).clamp(min=1)
            perp = (-(p * (p + 1e-10).log()).sum(dim=-1)).exp()
            util_str = ' '.join(f'{u:.2f}' for u in util.tolist())
            perp_str = ' '.join(f'{v:.0f}' for v in perp.tolist())
            disc_str = (f' | adv={total_adv/n:.4f} | fm={total_fm/n:.4f} | d={total_d/n:.4f}'
                        if use_disc else '')
            print(
                f'epoch {epoch:4d} | '
                f'recon={total_recon/n:.4f} | '
                f'commit={total_commit/n:.4f}'
                f'{disc_str} | '
                f'util=[{util_str}] | '
                f'perp=[{perp_str}]'
            )

        if epoch % vis_every == 0:
            visualize(model, probe, epoch, raw_length=probe_raw_length)
            print(f'  -> saved vis/epoch_{epoch:04d}_{{traces,psd}}.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Train or inspect the EEG Encodec path.')
    parser.add_argument(
        '--dummy-arch-check',
        action='store_true',
        help='Run one synthetic EEG epoch through encoder, RVQ, and decoder, then print layer shapes.',
    )
    parser.add_argument('--dummy-channels', type=int, default=32)
    parser.add_argument('--dummy-length', type=int, default=768)
    parser.add_argument('--dummy-seed', type=int, default=0)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--max-files', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--disc-warmup', type=int, default=50)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--vis-every', type=int, default=50)
    parser.add_argument('--no-disc', action='store_true')
    parser.add_argument('--no-kmeans-init', action='store_true')
    parser.add_argument('--checkpoint', default='eeg_overfit.pt')
    args = parser.parse_args()

    if args.dummy_arch_check:
        run_dummy_arch_check(
            channels=args.dummy_channels,
            length=args.dummy_length,
            seed=args.dummy_seed,
        )
        return

    # Overfit on a tiny slice: first 2 files = ~128 epochs
    data_dir = resolve_data_dir(args.data_dir)
    dataset = EEGEpochDataset(data_dir, max_files=args.max_files)
    print(f'{len(dataset)} epochs loaded from {data_dir}')

    model = make_eeg_model(kmeans_init=not args.no_kmeans_init).to(DEVICE)
    disc = None if args.no_disc else make_eeg_discriminator().to(DEVICE)
    disc_params = sum(p.numel() for p in disc.parameters()) if disc is not None else 0
    print(
        f'generator {sum(p.numel() for p in model.parameters()):,} params | '
        f'discriminator {disc_params:,} params'
    )
    print(
        f'frame_rate={model.frame_rate} fps | '
        f'bandwidth={model.bandwidth:.4f} kbps | '
        f'n_q={N_Q} codebooks | '
        f'disc warmup={args.disc_warmup} epochs'
    )

    train(
        model,
        dataset,
        disc=disc,
        n_epochs=args.epochs,
        lr=args.lr,
        disc_warmup=args.disc_warmup,
        log_every=args.log_every,
        vis_every=args.vis_every,
    )
    if args.checkpoint:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'sample_rate': SAMPLE_RATE,
                'ratios': RATIOS,
                'n_q': N_Q,
                'bandwidth': model.bandwidth,
                'data_dir': data_dir,
                'max_files': args.max_files,
                'epochs': args.epochs,
            },
            args.checkpoint,
        )
        print(f'saved checkpoint to {args.checkpoint}')


if __name__ == '__main__':
    main()
