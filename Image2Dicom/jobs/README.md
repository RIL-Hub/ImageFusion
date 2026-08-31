# jobs

`image2dicom probe` writes a job file here; you edit it, then `convert` runs it.
One file per image set, grouped by specimen.

The folders here are **worked examples** from real conversions. Their `input.path`
values point at scans that are not in this release, so they will not run as-is - read
them to see what a filled-in job file looks like, then write your own from `probe`.

Between them they cover every case you are likely to hit:

| | |
|---|---|
| `zeego_0001.yaml` | DICOM in, nothing to fill in, centred |
| `zeiss_bin2x2.yaml` | extensionless DICOM, same |
| `pet_plant.yaml` | TIFF in - spacing, modality and units all supplied, not centred |
| `pet_fiducials.yaml` | the same, for the fiducial scan |

Paths in a job file are relative to **the job file itself**, not to wherever you run
the command from.
