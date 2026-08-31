# ImageFusion

Register PET, Zeego CT and Zeiss microCT scans of the same specimen to one another, by picking matching landmarks by eye.

Two programs, run in order:

| | |
|---|---|
| **Image2Dicom** | Converts one image set — DICOM, `.IMA`, TIFF — into a uniform DICOM series. |
| **Aligner** | Opens those series together, takes your landmarks, and writes out registered copies. |

---

## Commands

Image2Dicom, run from `ImageFusion/Image2Dicom`:

```
python -m image2dicom probe <image-set> <job-file>          inspect one image set, write a job file
python -m image2dicom convert <job-file> [--dry-run]        run the conversion that job file describes
```

Aligner, run from `ImageFusion/Aligner`:

```
python -m aligner doctor                                    what is installed, and is napari where we expect it
python -m aligner inspect <series-dir> [...]                report geometry, open nothing
python -m aligner view <series-dir> [...] [--budget-mb N]   open them together and register them
python -m aligner solve <session-file>                      re-run the fits, report residuals
python -m aligner apply <session-file> <out-dir> [--overwrite]   write registered series and transforms
```

`<image-set>` is one file or one directory holding exactly one volume. `<series-dir>` is a directory Image2Dicom wrote. `<session-file>` is the YAML the viewer saves.

`solve` and `apply` import no Qt and no napari, so a session file replays the whole registration with the viewer uninstalled.

---

## Setup

One conda environment for both. Use the **conda-forge** channel for Qt — mixing it with pip's Qt is the one thing guaranteed to break the viewer.

```
conda create -n imagefusion -c conda-forge python=3.12
conda activate imagefusion
pip install "napari[all]>=0.9" "pydicom<3" numpy scipy "dask[array]" tifffile pyyaml typer rich
```

Check it:

```
python -m aligner doctor
```

Everything should read `[ok  ]`. If not, the line tells you what's missing.

---

## 1. Convert each image set

Two steps per image set: **probe** writes a job file describing what it found, you edit it, then **convert** does the work.

```
python -m image2dicom probe <image-set> <job-file>
python -m image2dicom convert <job-file>
```

For example:

```
python -m image2dicom probe "../../data/BEAN160519/Zeego/DYNACT_HEAD_NAT_FILL_HU_NORMAL_[INSPACE3D]_0001" jobs/BEAN160519/zeego.yaml
python -m image2dicom convert jobs/BEAN160519/zeego.yaml
```

Probe prints what it found and stops. Open the job file it wrote and:

- Set `output.directory` to where the series should go.
- Fill in anything marked **REQUIRED**. TIFF carries no spacing, modality or units, so PET inputs always need `spacing_mm`, `modality: "PT"` and `units`.
- Set `accept_warnings: true` to acknowledge what probe reported.
- Set `center: true` on the CTs. **Not on PET** — it is the reference and keeps its own coordinates.

Then convert. Add `--dry-run` first if you want to see the plan without writing anything.

---

## 2. Register

**Load order sets the chain** — each image registers to the one after it, and the last never moves.

```
python -m aligner view <series-dir> <series-dir> <series-dir>
```

For example:

```
python -m aligner view ../Image2Dicom/out/BEAN160519/zeiss ../Image2Dicom/out/BEAN160519/zeego ../Image2Dicom/out/BEAN160519/pet_fiducials
```

Add `--budget-mb N` to change how much memory each volume may use before it is downsampled to fit (default 1024).

**Orient.** Use the *orientation* panel to flip and rotate each volume until you can recognise features. This only affects what you see; it never changes the answer.

**Pick landmarks.** Select an image's `… landmarks` layer and use napari's point tool. Or place the crosshair with **Shift+click** (or **T**), fine-tune it in the *crosshair* panel, and press **add at crosshair** on that image's tab — the crosshair can be adjusted and checked from three directions before you commit it.

**Link them.** In the *landmarks* panel each image has its own tab. Every pick starts as its own landmark. Select one, choose another image under **link to**, type the number they should share, and press **link**. The *also in* column shows where each landmark already appears. **unlink** gives the selected pick its own number again, leaving its former partners alone; **delete** removes it from that image.

Each *link in the chain* needs at least **three** shared landmarks, not on a line. An image with no landmarks at all is simply passed over, so you can load a second PET scan purely to look at.

**Solve.** In the *registration* panel, read the report:

- **rms** is the headline accuracy, in mm.
- **worst** much larger than rms means one bad pick — find it and re-place it.
- A **measured scale** away from 1.0000 means a spacing problem in the conversion, not the registration.
- Warnings about collinear landmarks mean the fit looks good and isn't.

**Check it.** **show fit** moves everything on screen into the fixed image's frame. Look at whether the anatomy lines up, not just the marks. **undo fit** puts it back. Nothing is written either way.

**Save the session** before exporting. It holds your landmarks and is the one thing you can't regenerate.

---

## 3. Export

**export registered series…** in the *registration* panel, or:

```
python -m aligner apply <session-file> <out-dir>
```

For example:

```
python -m aligner apply session.yaml ../out/BEAN160519_registered
```

You get one directory per moving image plus `transforms.yaml`, which records each matrix alongside how well it fitted. The fixed image isn't copied — it never moved. Add `--overwrite` to replace series already written there.

To re-run a saved session without opening the viewer:

```
python -m aligner solve <session-file>
```

---

## Shortcuts

| | |
|---|---|
| `Shift+Z` / `Shift+Y` / `Shift+X` | look down that axis, keeping your place |
| `Shift+click`, or `T` | mark a point the views keep |

Shortcuts only work while the image has focus — after typing in a panel, click the image first.