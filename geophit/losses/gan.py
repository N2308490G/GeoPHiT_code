"""Re-export vanilla BCE GAN losses (Eq. 6 of the paper)."""

from ..models.discriminator import vanilla_d_loss, vanilla_g_loss

__all__ = ["vanilla_d_loss", "vanilla_g_loss"]
