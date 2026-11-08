# osmo360-reframe

A small set of Python scripts that turn DJI Osmo 360 `.OSV` files into stabilized, flat 4K (3840×2160) 50 fps video. You choose where to look over time; the scripts handle the stitching, stabilization and rendering on an NVIDIA GPU.

This is an unofficial personal project. It is **not affiliated with, endorsed by, or supported by DJI**. It does not use DJI Studio, Gyroflow, or any DJI software or SDK.

[日本語の説明は下にあります](#日本語)

## What it does

- Reads the camera's own per-frame orientation, which the Osmo 360 stores inside the `.OSV`, and uses it to stabilize (rotation only, 3 axes).
- Stitches the two fisheye lenses on the fly and cuts out a normal perspective view in the direction you specify.
- Renders 29.97 fps source to 50 fps output. With `blend`, each output frame mixes the two nearest source frames after each has been de-rotated, which avoids the "move, move, stop" judder of frame repetition.
- Writes HEVC Main 10 (`hvc1`), 3840×2160, 50 fps, bt709, timescale 50000, AAC 192 kbps stereo, matching Osmo Action footage so the two can be joined as-is.

## What is automatic and what is not

| Step | Who | Notes |
|---|---|---|
| Lens calibration (once per camera) | program | Fits 12 lens numbers by gradient descent. No trained model. |
| Gyro-to-camera axis and time alignment (once per camera) | program | Feature tracking + rotation fitting. Searches a clock offset of ±60 ms in 1 ms steps; the shipped file for the author's camera ended up at 2.4 ms (`dt_us: 2400`). |
| Extracting orientation from `.OSV` | program | Pure parsing, standard library only. |
| Overview contact sheets (`preview.py`) | program | Equirectangular tiles with yaw/pitch grid lines. |
| **Choosing angles (`plans.json`)** | **you, or an AI assistant** | No angle selection logic is in the code. |
| Stabilized view path | program | |
| Stitching, reframing, encoding | program | PyTorch on CUDA + ffmpeg (NVDEC/NVENC). |

None of the scripts call an AI service. PyTorch is used as a GPU array library, not to run a neural network. The author used an AI assistant to look at the contact sheets and write `plans.json`; you can write it by hand instead, or use a single fixed angle for the whole clip.

## Requirements

- An NVIDIA GPU with CUDA, plus hardware HEVC decode and HEVC Main 10 encode (NVDEC/NVENC).
- Python 3 with PyTorch (CUDA build), NumPy and Pillow, plus `opencv-python` for `rot360.py`, `jitter2.py` and `upright2.py`. `pip install -r requirements.txt` installs the Python packages (install the CUDA build of PyTorch first, see the note in that file).
- `ffmpeg` and `ffprobe` built with CUDA support (`-hwaccel cuda`, `scale_cuda`, `hevc_nvenc`).
- Developed and tested on Windows. `preview.py` and `preview_path.py` look for the Windows Arial font and fall back to a tiny built-in font if it is missing; edit the path at the top of those files on other systems.

The code assumes the recording settings it was built on: 3840×3840 per lens, 29.97 fps, and a 2048×1024 `.LRF` proxy file. Other modes need code changes.

Rough speed on the author's PC: about 70 seconds to render 20 seconds of output.

## Quick start

Run the commands from the repository folder. `.` is passed as the script folder to the scripts that need it.

### Once per camera (advanced)

This part exists because lens values differ slightly between units. To just try the pipeline you can use the shipped `calib_osmo360.json` and `gyrocal_osmo360.json` (copy them to `calib.json` / `gyrocal.json`); stitching seams and stabilization will be a little off on another camera.

1. Lens calibration. Put five pairs of same-moment frames from the two lenses in one folder, named `f0_0.png`, `f0_1.png`, … `f4_0.png`, `f4_1.png` (`_0` = video stream 0, `_1` = video stream 1). For example:

   ```sh
   ffmpeg -ss 10 -i path/to/sample.OSV -map 0:v:0 -frames:v 1 path/to/calib_frames/f0_0.png
   ffmpeg -ss 10 -i path/to/sample.OSV -map 0:v:1 -frames:v 1 path/to/calib_frames/f0_1.png
   # repeat with other times for f1 … f4
   python calib.py . path/to/calib_frames      # writes path/to/calib_frames/calib.json
   cp path/to/calib_frames/calib.json calib.json
   ```

2. Align the gyro with the camera axes using one sample clip:

   ```sh
   python osvmeta.py path/to/sample.OSV gyro_sample.json
   python rot360.py  calib.json path/to/sample.OSV rot_sample.json
   python gyrocmp.py gyro_sample.json rot_sample.json gyrocal.json
   ```

`calib_osmo360.json` and `gyrocal_osmo360.json` in this repository are the author's results for one camera.

### Per clip

```sh
python osvmeta.py   path/to/clip.OSV gyro_clip.json
python preview.py   . calib.json path/to/clip.LRF clip previews/ 2      # one tile every 2 s
# look at previews/ and write plans.json (below)
python stab360.py   gyro gyro_clip.json gyrocal.json plans.json clip stab_clip.json
python render360s.py . calib.json path/to/clip.OSV stab_clip.json out.mp4 - - blend
```

`render360s.py` arguments after `out.mp4` are `[t_end|-] [max_out_frames|-] [blend]`. `stab360.py` takes an optional last argument for the `cam` heading smoothing in seconds (default 0.8).

If you use one fixed angle, you can skip `preview.py`.

`sheet.py` is a separate quick-look tool, not part of the pipeline: `python sheet.py out.jpg a.OSV b.OSV ...` writes one row per file with three frames (at 10 %, 50 % and 90 % of the duration), taken from one lens only. It is not a panorama.

`jitter2.py` (`python jitter2.py <label> <start_s> <secs> <video> [<video> ...]`) measures residual shake of rendered videos, as used in the table below.

## Which files to use

| Stage | Current path | Legacy / experimental (kept for reference) |
|---|---|---|
| Orientation | `osvmeta.py` (reads the camera's own attitude) → `stab360.py gyro` | `rot360.py` + `upright2.py` + `stab360.py image` (image-based; it drifted: 146° of "up" error in 90 s in the author's test) |
| Rendering | `render360s.py` | `render360.py` (yaw-only correction, repeats frames for 30→50 fps) with `motion.py` |
| Checking an angle plan | none | `preview_path.py` (uses the old yaw-only path from `motion.py`, so it can differ from the stabilized render), `eqcheck.py` |
| Shared | `geom360.py`, `path360.py`, `gyrocmp.py` (imported by the scripts above) | |

The legacy scripts are not needed for the normal workflow. `rot360.py` is still used once per camera to align the gyro.

## plans.json

One list of keyframes per clip name. The clip name must match the `clip` argument of `stab360.py`. A copy-and-edit example is in `examples/plans.example.json`. The key is **`lock`** (not `mode`); leaving it out raises a `KeyError`.

```json
{
  "clip": [
    {"t": 0,    "lock": "world", "ref": 0.5,  "yaw": 60,  "pitch": 20,  "fov": 100},
    {"t": 5.5,  "ease": 1.5, "lock": "cam",   "yaw": -10, "pitch": -27, "fov": 105},
    {"t": 12.5, "ease": 1.5, "lock": "world", "ref": 12.5, "yaw": 110, "pitch": 15, "fov": 100}
  ]
}
```

| Key | Meaning |
|---|---|
| `t` | Time (s) at which this angle is reached |
| `lock` | `"world"`: fixed to the scene. `"cam"`: not rigidly fixed to the camera. It follows the camera's left-right heading smoothed with a Gaussian of about 0.8 s (last argument of `stab360.py`), so fast hand shake is ignored but slow turns are followed. Pitch stays referenced to the horizon in both modes |
| `yaw` | Degrees. 0 = lens 0 axis, positive = to the right. Read it from the yellow lines on the contact sheet |
| `pitch` | Degrees. Positive = up, 0 = horizon |
| `fov` | Horizontal field of view in degrees (90–110 is typical) |
| `ref` | `world` only: the time of the contact sheet you read `yaw` from. Defaults to `t` |
| `ease` | Seconds to move from the previous angle, ending at `t` (smoothstep). Default 1.5 |
| `note` | Free text, ignored |

The contact sheets are in camera coordinates and are not leveled, while rendering uses horizon-referenced pitch. If the camera was held tilted, expect some pitch offset.

## Measured shake

`jitter2.py` measures the median whole-frame motion in 4K pixels over stretches with a constant angle. On two test clips (six stretches):

| | before (old renderer) | after (gyro + blend) | Osmo Action, for reference |
|---|---|---|---|
| Left-right shake (faster than 0.4 s) | 17–34 px | 4.7–9.5 px | 22–27 px |
| Up-down shake | 18–40 px | 2.1–4.6 px | 9–15 px |
| Frame-to-frame unevenness | 5.2–9.5 | 0.4–0.9 | 0.7–1.0 |

Computed from the table: left-right shake is about 3–6 times smaller than with the old renderer, up-down shake about 8–15 times smaller; both are below the Osmo Action reference. Do not measure stretches that contain an angle change (`ease`); the numbers jump there.

## Limitations

- Detail per degree is lower than a normal camera (about 20 px per degree), so 4K output is slightly soft.
- Only rotation is removed. Translation from walking remains and is most visible on nearby subjects.
- Fast camera swings are motion-blurred in the source; stabilization cannot remove that.
- `blend` can leave a faint double image on fast-moving nearby subjects. Render without `blend` if that bothers you (judder returns).
- No automatic highlight or subject selection.
- Audio is plain stereo; no spatial audio.
- The `.LRF` proxy is needed for the contact sheets.
- `preview_path.py` still uses the older image-based path (`motion.py`), so its preview can differ slightly from the stabilized render.
- Sizes and frame rates of the supported recording mode are hard-coded in the scripts (see Requirements).

## Credits

`osvmeta.py` was written from the format notes and protobuf definitions published by the OpenOSV project (Apache-2.0). No OpenOSV code is included or executed. The scripts were written with the help of an AI coding assistant.

## License

MIT, see `LICENSE`.

---

<a id="日本語"></a>

# 日本語

DJI Osmo 360 の `.OSV` ファイルを、揺れを止めた横長の 4K（3840×2160）・50fps の動画にする Python スクリプト集。どの方向を映すかを決めると、2つのレンズのつなぎ合わせ・揺れ止め・書き出しを NVIDIA の GPU で行う。

個人が作った非公式の道具で、**DJI とは関係ない**。DJI Studio・Gyroflow など DJI のソフトや開発キットは使っていない。

## できること

- Osmo 360 が `.OSV` の中に毎コマ書き込んでいる「カメラの向き」の記録を読み、揺れ止めに使う（回転の3軸）。
- 2つの魚眼レンズの絵をその場でつなぎ、指定した方向をふつうの四角い画面で切り出す。
- 1秒29.97枚の元映像から50枚を作る。`blend` を付けると、近い2枚をそれぞれ向きを打ち消してから混ぜるので、同じ絵の繰り返しによるカクつきが出ない。
- HEVC Main 10（`hvc1`）・3840×2160・50fps・bt709・タイムスケール 50000・AAC 192kbps ステレオで保存する。Osmo Action の動画とそのままつなげられる。

## 自動の所と、人か AI が決める所

アングル選び（`plans.json` を書くこと）以外は、すべてプログラムの計算。スクリプトの中で AI を呼び出す所はない（揺れ止めと書き出しは決まった手順の計算）。PyTorch は GPU で大量の計算を同時にするために使っていて、AI のモデルは入っていない。作者はアングル選びに AI アシスタントを使ったが、手で書いてもいいし、クリップ全体を1つのアングルにしてもいい。

## 必要なもの

- CUDA が使える NVIDIA の GPU（HEVC の読み込みと 10bit の書き出しができるもの）
- Python 3、PyTorch（CUDA 版）、NumPy、Pillow
- `opencv-python`（一度だけの `rot360.py` と、測定用の `jitter2.py`・`upright2.py` で使う）
- CUDA 対応の `ffmpeg` と `ffprobe`
- Windows で作って確かめた。`preview.py` と `preview_path.py` は Windows のフォント（Arial）の場所を直接書いているので、ほかの OS では直す

撮影設定は、1レンズ 3840×3840・29.97fps・代理ファイル `.LRF` 2048×1024 を前提にしている。

## 使い方

コマンドは上の英語の「Quick start」と同じ。流れは次のとおり。

1. カメラ1台につき一度：`calib.py` でレンズを合わせ、`osvmeta.py`・`rot360.py`・`gyrocmp.py` でセンサーの軸と時刻を合わせる。同梱の `calib_osmo360.json`・`gyrocal_osmo360.json` は作者のカメラ1台の値。試すだけならそのまま使えるが、カメラごとにレンズの値が少し違うので、ちゃんと使うなら自分のカメラでやり直す（`calib.py` には、同じ瞬間の2レンズの絵を5組、`f0_0.png`〜`f4_1.png` の名前で用意する。上級者向け）。
2. クリップごと：`osvmeta.py` → `preview.py`（一覧画像）→ `plans.json` を書く → `stab360.py` → `render360s.py`（最後に `blend`）。

`plans.json` の書き方は英語の「plans.json」の表と同じ（例は `examples/plans.example.json`）。キー名は `lock`（`mode` ではない。書かないとエラーになる）。`"world"` は景色に固定。`"cam"` はカメラに固定ではなく、左右の向きだけを約0.8秒の幅でならしてついていく（手の細かい揺れは追わず、ゆっくりした向き変えに追従する）。上下は両方とも水平が基準。

揺れ止め版の道筋は `osvmeta.py → stab360.py gyro → render360s.py（最後に blend）`。`render360.py`・`motion.py`・`preview_path.py` などは前の版で、ふだんは使わない。`preview_path.py` は前の計算なので、揺れ止め版の書き出しと少し違う。`sheet.py` は全景ではなく、ファイルごとに 10%・50%・90% の3コマ（片方のレンズだけ）を並べる簡易確認用。

## 測った揺れ

2本の試しクリップ（6区間）で、揺れは左右 17〜34 → 4.7〜9.5 ピクセル（約3〜6分の1）、上下 18〜40 → 2.1〜4.6 ピクセル（約8〜15分の1）。コマごとのむらは 5.2〜9.5 → 0.4〜0.9。

## できないこと

- 1度あたり約20画素なので、4K にすると少し甘い。
- 消せるのは回転だけ。歩いて位置が動く揺れは残る。
- 素早く振った場面のぶれは消せない。
- `blend` で、近くを素早く動く物にうっすら二重の影が出ることがある。
- 見どころの自動選択、立体音響はない。

## 仕組みの説明

- [仕組みの説明（Markdown）](docs/HOW_IT_WORKS.md)
- [仕組みの説明（スマホ向けの1ページ）](docs/how-it-works.html)

## クレジット

`osvmeta.py` は OpenOSV（Apache-2.0）が公開しているファイル形式の説明と protobuf の定義を読んで書いた。OpenOSV のコードは含めていないし、実行もしていない。スクリプトは AI のコーディング支援を使って書いた。

## ライセンス

MIT（`LICENSE` を参照）。
