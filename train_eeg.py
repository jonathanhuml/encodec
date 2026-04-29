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

def make_eeg_model() -> EncodecModel:
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
        kmeans_init=True,
        kmeans_iters=50,
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.epochs[idx]  # [C, T]

        # pad T to a multiple of the encoder's total stride
        T = x.shape[-1]
        rem = T % TOTAL_STRIDE
        if rem:
            x = F.pad(x, (0, TOTAL_STRIDE - rem))

        # per-channel z-score so every electrode is on the same scale
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
        return (x - mu) / sigma  # [C, T_padded]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

N_TRACE_CHANNELS = 6   # channels to show in time-trace plot
VIS_DIR = 'vis'


def visualize(model: EncodecModel, x: torch.Tensor, epoch: int) -> None:
    """
    x: [C, T] on device (already z-scored).
    Saves two figures per call:
      vis/epoch_{n}_traces.png  — original vs reconstructed time traces
      vis/epoch_{n}_psd.png     — mean PSD original vs reconstructed
    """
    os.makedirs(VIS_DIR, exist_ok=True)
    model.eval()
    with torch.no_grad():
        recon, loss_recon, _, _codes = forward_eeg(model, x)
    model.train()

    orig = x.cpu().numpy()       # [C, T]
    rec  = recon.cpu().numpy()   # [C, T]
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

def forward_eeg(model: EncodecModel, x: torch.Tensor):
    """
    x: [C, T] — one epoch on device.
    Returns (recon [C, T], loss_recon, loss_commit, codes [n_q, C, T_frames]).
    """
    C, T = x.shape
    x_flat = x.unsqueeze(1)                                    # [C, 1, T]

    z = model.encoder(x_flat)                                  # [C, 128, T//16]
    qres = model.quantizer(z, model.frame_rate, model.bandwidth)
    recon = model.decoder(qres.quantized)                      # [C, 1, T]

    loss_recon = F.mse_loss(recon, x_flat)
    loss_commit = qres.penalty
    return recon.squeeze(1), loss_recon, loss_commit, qres.codes


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

    probe = dataset[0].to(DEVICE)

    model.train()
    if disc is not None:
        disc.train()

    for epoch in range(1, n_epochs + 1):
        use_disc = disc is not None and epoch > disc_warmup
        total_recon = total_commit = total_adv = total_fm = total_d = 0.0
        code_counts = torch.zeros(N_Q, 1024)

        for x in loader:
            x = x.to(DEVICE)
            recon, loss_recon, loss_commit, codes = forward_eeg(model, x)

            x_d = x.unsqueeze(1)      # [C, 1, T]
            r_d = recon.unsqueeze(1)  # [C, 1, T]

            # --- generator step ---
            if use_disc and x.shape[-1] >= DISC_MIN_T:
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
            if use_disc and x.shape[-1] >= DISC_MIN_T:
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
            visualize(model, probe, epoch)
            print(f'  -> saved vis/epoch_{epoch:04d}_{{traces,psd}}.png')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Overfit on a tiny slice: first 2 files = ~128 epochs
    dataset = EEGEpochDataset(DATA_DIR, max_files=2)
    print(f'{len(dataset)} epochs loaded from {DATA_DIR}')

    model = make_eeg_model().to(DEVICE)
    disc  = make_eeg_discriminator().to(DEVICE)
    print(
        f'generator {sum(p.numel() for p in model.parameters()):,} params | '
        f'discriminator {sum(p.numel() for p in disc.parameters()):,} params'
    )
    print(
        f'frame_rate={model.frame_rate} fps | '
        f'bandwidth={model.bandwidth:.4f} kbps | '
        f'n_q={N_Q} codebooks | '
        f'disc warmup=50 epochs'
    )

    train(model, dataset, disc=disc)
