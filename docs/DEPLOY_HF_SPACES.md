# Deploying the demo to Hugging Face Spaces

The repo ships a Gradio app (`app.py`) that runs the full pipeline on any
uploaded image. Deploying it to a free Hugging Face Space gives you a public
URL anyone can try.

## Steps

1. **Create the Space** — https://huggingface.co/new-space, SDK: **Gradio**,
   hardware: CPU basic (free) is enough.

2. **Prepare requirements** — Spaces installs from `requirements.txt` at the
   repo root. Uncomment the demo lines so it reads:

   ```
   torch>=2.2.0
   torchvision>=0.17.0
   opencv-python-headless>=4.10.0   # use -headless on Spaces
   pillow>=10.0.0
   numpy>=1.26.4
   pyyaml>=6.0.1
   tqdm>=4.66.5
   timm>=0.9        # MiDaS monocular depth
   gradio>=4.0
   ```

   (Swap `opencv-python` for `opencv-python-headless` — Spaces has no display
   server.)

3. **Include a trained detector** — copy `models/detector_best.pt` into the
   Space repo (it is gitignored here; use `git add -f models/detector_best.pt`
   in the Space clone, or upload via the web UI). Optionally add
   `models/cmSalGAN.ckpt` for full cmSalGAN saliency; without it the demo
   uses the labeled spectral-residual fallback.

4. **Push**:

   ```bash
   git clone https://huggingface.co/spaces/<you>/<space-name>
   # copy the project files in (app.py must be at the root)
   git add . && git commit -m "Deploy saliency driving demo" && git push
   ```

5. The Space builds automatically; first startup downloads the MiDaS weights
   (~66 MB, cached afterwards).

## Notes

- `app.py` lazy-loads all models on the first request, so the Space boots fast.
- README metadata (optional): add a YAML header to the Space's README.md with
  `sdk: gradio` and `app_file: app.py` if you don't use the web wizard.
- Free CPU hardware runs Faster R-CNN in ~5 s/image — acceptable for a demo.
