import torch


def get_white_noise(
    signal_embedding: torch.Tensor, signal_to_noise_ratio: float
) -> torch.Tensor:
    rms_signal = torch.sqrt(torch.mean(signal_embedding**2))

    rms_noise = torch.sqrt(
        rms_signal**2
        / torch.pow(torch.tensor(10), signal_to_noise_ratio / 10)
    )

    std_noise = rms_noise
    size = tuple(signal_embedding.shape)
    return torch.normal(0.0, std_noise.cpu().item(), (size[-2], size[-1]))
