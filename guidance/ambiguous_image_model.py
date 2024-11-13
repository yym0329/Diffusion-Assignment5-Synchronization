from abc import *
from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
import os
import json

from guidance.base_model import BaseModel
from utils.image_utils import merge_images
from utils.views import get_views


from abc import *
from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from PIL import Image
import os
import json


def tensor_to_pil(x):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.clone().detach()
    x = (
        (x / 2 + 0.5)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", torch.uint8)
        .numpy()
    )
    images = [Image.fromarray(xx) for xx in x]
    if len(images) == 1:
        return images[0]
    else:
        return images


def decode_latent(vae, latent):
    x = vae.decode(latent / vae.config.scaling_factor).sample
    return x


class AmbiguousImageModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.gpu)

        super().__init__()
        self.initialize()

    def initialize(self):
        super().initialize()

        self.num_prompts = len(self.config.prompts)
        self.intermediate_dir = os.path.join(self.output_dir, "intermediate")
        self.result_dir = os.path.join(self.output_dir, "results")

        for _dir in [self.intermediate_dir, self.result_dir]:
            os.makedirs(_dir, exist_ok=True)

        log_opt = vars(self.config)
        config_path = os.path.join(self.output_dir, "ambiguous_image_config.yaml")
        with open(config_path, "w") as f:
            json.dump(log_opt, f, indent=4)

    def init_mapper(self, views_names: Optional[List] = None):
        if views_names is None:
            views_names = self.config.views_names
        self.config.views_names = views_names
        print(f"[*] Mappers: {views_names}")
        self.views = get_views(views_names, self.config.rotate_angle)

    def init_prompt_embeddings(self, prompts: Optional[List] = None):
        if prompts is None:
            prompts = self.config.prompts
        self.config.prompts = prompts
        print(f"[*] Prompts: {prompts}")
        self.prompt_embeds = self.compute_prompt_embeds(prompts)

    def compute_prompt_embeds(self, prompts: Optional[List[str]] = None):
        """
        prompts: List [N]
        prompt_embeds: [2N,L,D]
        """
        if prompts is None:
            prompts = self.config.prompts

        prompt_embeds = [
            self.stage_1.encode_prompt(
                f"{p}".strip(),
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
                device=self.device,
            )
            for p in prompts
        ]

        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])  # [2N,L,D]

        return prompt_embeds

    def initialize_latent(
        self, latent_space: Literal["instance", "canonical"], **kwargs
    ):
        generator = kwargs.get("generator")
        model = kwargs.get("model")
        device = self.device
        num_channels = kwargs.get("num_channels")
        if num_channels is None:
            num_channels = (
                model.unet.config.in_channels // 2
                if model == getattr(self, "stage_2", None)
                else model.unet.config.in_channels
            )
        height = model.unet.config.sample_size
        width = model.unet.config.sample_size
        num_prompts = self.num_prompts

        latents = model.prepare_intermediate_images(
            num_prompts,
            num_channels,
            height,
            width,
            self.prompt_embeds.dtype,
            device,
            generator,
        )  # [num_prompts,3,64,64]

        return latents

    def forward_ft(self, canonical_input, index, **kwargs):
        view_ft = self.views[index]
        instance_out = view_ft.view(canonical_input)
        return instance_out

    def inverse_ft(self, screen_input, index, **kwargs):
        view_ft = self.views[index]
        canonical_output = view_ft.inverse_view(screen_input)
        return canonical_output

    def forward_mapping(self, z_t, **kwargs):
        """
        z_t: [1,C,H,W]
        x_t: [N,C,H,W]
        """

        z_t = z_t.squeeze(0)
        num_views = len(self.views)
        x_ts = [self.forward_ft(z_t, i, **kwargs) for i in range(num_views)]
        x_ts = torch.stack(x_ts, 0)

        return x_ts

    def get_variable(self, var_type):
        return getattr(self, f"_{var_type}")

    def set_variable(self, var_type, input):
        return setattr(self, f"_{var_type}", input)

    @torch.enable_grad()
    def inverse_mapping(self, x_ts, **kwargs):
        """
        x_ts: [N,C,H,W]
        z_t: [1,C,H,W]
        """

        num_views = len(self.views)
        z_ts = [self.inverse_ft(x_ts[i], i, **kwargs) for i in range(num_views)]
        z_ts = torch.stack(z_ts, 0)
        z_t = torch.mean(z_ts, 0)
        z_t = z_t.unsqueeze(0)

        return z_t

    def compute_noise_preds(self, xts, ts, **kwargs):
        """
        Input:
            xts: [N,C,H,W]
            ts: [1]
        Output:
            noise_preds: [N,C,H,W]
        """
        stage_2 = kwargs.get("stage_2")
        if stage_2:
            orig_xts_shape = xts.shape
            C, H, W = xts.shape[-3], xts.shape[-2], xts.shape[-1]
            xts = xts.reshape(-1, C, H, W)

            upscaled = kwargs.get("upscaled")
            noise_level = kwargs.get("noise_level")
            model_input = torch.cat([xts, upscaled], dim=1)
            model_input = torch.cat([model_input] * 2)
            model_input = self.stage_2.scheduler.scale_model_input(model_input, ts)

            noise_preds = self.stage_2.unet(
                model_input,
                ts,
                encoder_hidden_states=self.prompt_embeds,
                class_labels=noise_level,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            noise_preds_uncond, noise_preds_text = noise_preds.chunk(2)
            noise_preds = noise_preds_uncond + self.config.guidance_scale * (
                noise_preds_text - noise_preds_uncond
            )
            noise_preds = noise_preds.reshape(*orig_xts_shape[:-3], -1, H, W)

            return noise_preds

        else:
            orig_xts_shape = xts.shape
            C, H, W = xts.shape[-3], xts.shape[-2], xts.shape[-1]
            xts = xts.reshape(-1, C, H, W)

            xts_input = torch.cat([xts] * 2)
            xts_input = self.stage_1.scheduler.scale_model_input(xts_input, ts)

            noise_preds = self.stage_1.unet(
                xts_input,
                ts,
                encoder_hidden_states=self.prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_preds_uncond, noise_preds_text = noise_preds.chunk(2)
            noise_preds = noise_preds_uncond + self.config.guidance_scale * (
                noise_preds_text - noise_preds_uncond
            )
            noise_preds = noise_preds.reshape(*orig_xts_shape[:-3], -1, H, W)

            return noise_preds

    @torch.no_grad()
    def __call__(self, tag=None, prompts=None, views_names=None, save_dir_now=True):
        self.init_prompt_embeddings(prompts=prompts)
        self.init_mapper(views_names=views_names)
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        self.stage_1.scheduler.set_timesteps(
            self.config.num_inference_steps, device=self.device
        )

        xts = self.initialize_latent(
            "instance", generator=generator, model=self.stage_1
        )
        zts = None

        input_params = {"zts": zts, "xts": xts}
        timesteps = self.stage_1.scheduler.timesteps

        alphas = torch.sqrt(self.stage_1.scheduler.alphas_cumprod).to(self.device)
        sigmas = torch.sqrt(1 - self.stage_1.scheduler.alphas_cumprod).to(self.device)
        for i, t in enumerate(timesteps):
            out_params = self.one_step_process(
                input_params,
                t,
                alphas,
                sigmas,
            )

            input_params["xts"] = out_params["x_t_1"]
            input_params["zts"] = None
            log_x_prevs = out_params["x_t_1"]
            log_x0s = out_params["x0s"]

            """ Logging """
            if (i + 1) % self.config.log_step == 0:
                log_x_prev_imgs = self.tensor_to_pil_img(log_x_prevs)
                log_x0_imgs = self.tensor_to_pil_img(log_x0s)

                log_img = merge_images([log_x_prev_imgs, log_x0_imgs])
                log_img.save(f"{self.intermediate_dir}/i={i}_t={t}.png")

        final_denoised = self.forward_mapping(self.inverse_mapping(out_params["x_t_1"]))
        final_denoised_imgs = self.tensor_to_pil_img(final_denoised)
        if type(final_denoised_imgs) != list:
            final_denoised_imgs = [final_denoised_imgs]
        for i, img in enumerate(final_denoised_imgs):
            img.save(f"{self.result_dir}/view_{i}.png")
        merge_images(final_denoised_imgs).save(f"{self.result_dir}/final.png")

        # Stage 2
        xts = self.initialize_latent(
            "instance", generator=generator, model=self.stage_2
        )
        zts = None

        previous_stage_imgs = final_denoised
        input_params = {"zts": zts, "xts": xts}

        height = self.stage_2.unet.config.sample_size
        width = self.stage_2.unet.config.sample_size

        image = self.stage_2.preprocess_image(previous_stage_imgs, 1, self.device)
        upscaled = F.interpolate(
            image, (height, width), mode="bilinear", align_corners=True
        )

        noise_level = torch.tensor(
            [self.config.stage2_noise_level] * 2 * upscaled.shape[0],
            device=self.device,
        )

        for i, t in enumerate(timesteps):
            out_params = self.one_step_process(
                input_params,
                t,
                alphas,
                sigmas,
                stage_2=True,
                upscaled=upscaled,
                noise_level=noise_level,
            )

            input_params["xts"] = out_params["x_t_1"]
            input_params["zts"] = None
            log_x_prevs = out_params["x_t_1"]
            log_x0s = out_params["x0s"]

            """ Logging """
            if (i + 1) % self.config.log_step == 0:
                log_x_prev_imgs = self.tensor_to_pil_img(log_x_prevs)
                log_x0_imgs = self.tensor_to_pil_img(log_x0s)

                log_img = merge_images([log_x_prev_imgs, log_x0_imgs])
                log_img.save(f"{self.intermediate_dir}/second_i={i}_t={t}.png")

        final_denoised = self.forward_mapping(self.inverse_mapping(out_params["x_t_1"]))
        final_denoised_imgs = self.tensor_to_pil_img(final_denoised)

        if type(final_denoised_imgs) != list:
            final_denoised_imgs = [final_denoised_imgs]

        for i, img in enumerate(final_denoised_imgs):
            img.save(
                f"{self.result_dir}/{self.config.prompts[i].replace(' ', '_')}.png"
            )
        merge_images(final_denoised_imgs).save(f"{self.result_dir}/final.png")

    @torch.no_grad()
    def tensor_to_pil_img(self, x) -> Union[Image.Image, List[Image.Image]]:
        """
        Input:
            x: [*,C,H,W]
        Output:
            img or List of imgs
        """
        C, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.reshape(-1, C, H, W)

        imgs = tensor_to_pil(x)
        return imgs
