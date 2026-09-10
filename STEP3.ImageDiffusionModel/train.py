from dataset.dataloader import get_loader
import time
from ddpm.unet import UNet
import os
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from ddpm import Unet3D, GaussianDiffusion, Trainer, Unet3D_CA, TUMOR_COLUMNS
from re import I
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.getcwd())


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.unet_dim,
            dim_mults=cfg.model.dim_mults,
            # image (1) and tumor mask (1)
            channels=cfg.model.diffusion_num_channels,
            out_dim=cfg.model.out_dim,
            num_continuous_conditioners=len(TUMOR_COLUMNS),
            num_organs=9
        ).cuda()
    elif cfg.model.denoising_fn == 'Unet3D_CA':
        x_channels = cfg.model.out_dim
        cond_channels = cfg.model.diffusion_num_channels - cfg.model.out_dim

        model = Unet3D_CA(
            dim=cfg.model.unet_dim,
            dim_mults=cfg.model.dim_mults,
            channels=x_channels,
            out_dim=cfg.model.out_dim,
            num_continuous_conditioners=len(TUMOR_COLUMNS),
            num_organs=9,
            cond_channels=cond_channels,
            num_res_blocks=2,
            attention_resolutions=(2, 4, 8),
            num_heads=8,
            # dim_head removed -- now computed internally as ch // num_heads
            # at every resolution level, matching source's legacy=True behavior
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        spatial_weight_loss=True,
        tumor_weight=1000.0
    ).cuda()

    val_dataset_cfg = OmegaConf.merge(
        cfg.dataset,
        cfg.val_dataset,
    )

    train_dataloader, train_sampler, dataset_size = get_loader(cfg.dataset)

    val_dataloader, _, _ = get_loader(val_dataset_cfg)

    # val_dataloader=None

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataloader,
        val_dataset=val_dataloader,
        train_batch_size=cfg.model.batch_size,
        save_and_sample_every=cfg.model.save_and_sample_every,
        validate_every=cfg.model.get('validate_every', 1000),
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        max_grad_norm=2.0,
        spatial_weight_loss=True,
        start_weight=1000.0,
        end_weight=1000.0,
        warmup_steps=0
    )

    if cfg.model.load_milestone:
        trainer.load(-1)  # load the latest checkpoint

    trainer.train()


if __name__ == '__main__':
    run()
