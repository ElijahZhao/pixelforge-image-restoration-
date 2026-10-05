import Link from "next/link";

export const metadata = {
  title: "Method | PixelForge",
};

export default function MethodPage() {
  return (
    <main className="mx-auto max-w-3xl px-5 py-10">
      <Link href="/" className="text-sky-300 text-sm hover:underline">
        ← Back
      </Link>
      <h1 className="text-3xl font-bold mt-4">Method &amp; Models</h1>
      <p className="mt-3 text-slate-300">
        PixelForge is an end-to-end computer-vision project: I implemented and
        trained the models myself in PyTorch, then deployed them behind a web
        interface. No black-box APIs: the full pipeline
        (data → training → evaluation → serving) is reproducible from this repo.
      </p>

      <h2 className="text-xl font-semibold mt-8">Tasks</h2>
      <ul className="mt-3 space-y-3 text-slate-300 list-disc pl-5">
        <li>
          <b>Single Image Super-Resolution (SR).</b> Upscales a low-resolution
          image by 2× or 4×. Implemented a classic{" "}
          <a
            className="text-sky-300"
            href="https://arxiv.org/abs/1501.00092"
          >
            SRCNN
          </a>{" "}
          baseline and a deeper SRResNet-style generator trained with a
          perceptual (VGG) loss for sharper results.
        </li>
        <li>
          <b>Low-Light Image Enhancement (LLIE).</b> Brightens dark photos while
          preserving structure, using a small U-Net trained on paired low/high
          images (LOL dataset).
        </li>
      </ul>

      <h2 className="text-xl font-semibold mt-8">Architectures</h2>
      <pre className="mt-3 glass p-4 text-xs overflow-x-auto text-slate-300">{`SRCNN       : Conv(9) -> ReLU -> Conv(5) -> ReLU -> Conv(5)
SRResNet    : Conv -> [ResBlock x8] -> Conv -> PixelShuffle(scale) -> Conv
LowLight UNet: Enc(3->32->64->128) -> Bottleneck -> Dec (skip connections)
                + residual learning (predicts an enhancement residual)`}</pre>

      <h2 className="text-xl font-semibold mt-8">Evaluation</h2>
      <p className="mt-3 text-slate-300">
        Quantitative results use PSNR and SSIM. Rows marked <b>ours</b> are
        filled with metrics measured on this repo&rsquo;s trained weights
        (SR&times;4 on DIV2K, low-light on LOL, full-image protocol); the
        SRCNN&times;2 row stays <b>TBD</b> because no weight was trained/exported
        for it (it falls back to the classical baseline). No placeholder numbers
        are ever shown as real results.
      </p>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm glass">
          <thead>
            <tr className="text-left text-slate-400">
              <th className="p-2">Method</th>
              <th className="p-2">Task</th>
              <th className="p-2">Scale</th>
              <th className="p-2">PSNR</th>
              <th className="p-2">SSIM</th>
            </tr>
          </thead>
          <tbody className="text-slate-200">
            <tr>
              <td className="p-2">Bicubic (baseline)</td>
              <td className="p-2">SR</td>
              <td className="p-2">2×</td>
              <td className="p-2">33.66 *</td>
              <td className="p-2">0.9299 *</td>
            </tr>
            <tr>
              <td className="p-2">SRCNN (ours)</td>
              <td className="p-2">SR</td>
              <td className="p-2">2×</td>
              <td className="p-2 text-slate-500">TBD</td>
              <td className="p-2 text-slate-500">TBD</td>
            </tr>
            <tr>
              <td className="p-2">Bicubic (baseline)</td>
              <td className="p-2">SR</td>
              <td className="p-2">4×</td>
              <td className="p-2">26.69</td>
              <td className="p-2">0.754</td>
            </tr>
            <tr>
              <td className="p-2">SRResNet (ours)</td>
              <td className="p-2">SR</td>
              <td className="p-2">4×</td>
              <td className="p-2">27.47</td>
              <td className="p-2">0.780</td>
            </tr>
            <tr>
              <td className="p-2">No-op (baseline)</td>
              <td className="p-2">LowLight</td>
              <td className="p-2">N/A</td>
              <td className="p-2">7.77</td>
              <td className="p-2">0.192</td>
            </tr>
            <tr>
              <td className="p-2">LowLight-UNet (ours)</td>
              <td className="p-2">LowLight</td>
              <td className="p-2">N/A</td>
              <td className="p-2">18.18</td>
              <td className="p-2">0.739</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        * Set5 ×2 Bicubic reference value (Dong et al.). The ×4 / low-light rows
        are measured in this repo (DIV2K / LOL, full-image protocol); SRCNN ×2
        remains TBD (no trained weight; it uses the classical fallback).
      </p>

      <h2 className="text-xl font-semibold mt-8">Datasets</h2>
      <ul className="mt-3 space-y-2 text-slate-300 list-disc pl-5 text-sm">
        <li>
          <b>DIV2K</b>: 800 high-quality training images for SR (LR synthesised
          by bicubic downscaling).
        </li>
        <li>
          <b>LOL-v1</b>: 500 paired low/high-light images for low-light
          enhancement.
        </li>
        <li>
          <b>Set5 / Set14</b>: standard SR test sets for evaluation.
        </li>
      </ul>

      <h2 className="text-xl font-semibold mt-8">Training</h2>
      <p className="mt-3 text-slate-300 text-sm">
        Trained on free GPUs (Kaggle / Colab). Example:
      </p>
      <pre className="mt-3 glass p-4 text-xs overflow-x-auto text-slate-300">{`python train/train.py --task sr --model generator --scale 4 \\
    --data_root data --epochs 200 --batch_size 8 --lr 1e-4 --perceptual
python train/export.py --checkpoint models/sr_generator_scale4_best.pth \\
    --out serve/models/sr_generator_scale4.pt --task sr --scale 4

# low-light (default --model srcnn --scale 2)
python train/train.py --task lowlight --data_root data --epochs 200
python train/export.py --checkpoint models/lowlight_srcnn_scale2_best.pth \\
    --out serve/models/lowlight.pt --task lowlight --scale 2`}</pre>

      <h2 className="text-xl font-semibold mt-8">Deployment</h2>
      <p className="mt-3 text-slate-300 text-sm">
        Frontend: Next.js (Vercel). Inference: FastAPI serving the exported
        TorchScript models (Hugging Face Spaces / small VPS). The service
        automatically falls back to classical baselines when no trained weights
        are present, so the demo always runs.
      </p>
    </main>
  );
}
