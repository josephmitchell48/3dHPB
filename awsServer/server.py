from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import os, subprocess, shlex, uuid, json, time, shutil

IN_ROOT, OUT_ROOT = "/tmp/in", "/tmp/out"
os.makedirs(IN_ROOT, exist_ok=True); os.makedirs(OUT_ROOT, exist_ok=True)

app = FastAPI(title="HPB Segmentation API")

def run(cmd: str, env=None):
    print("[cmd]", cmd, flush=True)
    env = (env or os.environ).copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, env=env)
    if res.returncode != 0:
        # Print stdout/stderr so they hit your EC2 console logs
        print("---- STDOUT ----\n", res.stdout, sep="", flush=True)
        print("---- STDERR ----\n", res.stderr, sep="", flush=True)
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return res

def nnunet_v1_task008(in_dir: str, out_dir: str, *, case_id: str, folds: str = "0") -> str:
    """
    Run nnU-Net v1 Task008 on a single case.
    Requires case_id (the basename WITHOUT _0000), so we can assert the expected output.
    Returns the absolute path to <out_dir>/<case_id>.nii.gz
    """
    env = os.environ.copy()
    env.setdefault("RESULTS_FOLDER", "/models/results_v1")
    cmd = (
        f"nnUNet_predict -i {in_dir} -o {out_dir} "
        f"-t Task008_HepaticVessel -m 3d_fullres -f {folds} "
        f"--disable_tta --num_threads_preprocessing 1 --num_threads_nifti_save 1 "
        f"-chk model_final_checkpoint"
    )
    run(cmd, env=env)

    expected = os.path.join(out_dir, f"{case_id}.nii.gz")
    if os.path.exists(expected):
        return expected

    # Safety fallback (should not be needed, but helps if nnUNet changes naming)
    for fn in os.listdir(out_dir):
        if fn.endswith(".nii.gz") and fn not in ("plans.pkl", "postprocessing.json"):
            return os.path.join(out_dir, fn)

    raise RuntimeError(f"Task008: expected output not found: {expected}")

def totalseg_liver_only(in_path: str, out_dir: str, fast: bool = False) -> str:
    """
    Predictable TS liver mask using roi_subset.
    Output is always <out_dir>/liver.nii.gz if successful.
    """
    flags = ["--fast"] if fast else []
    flag_str = " ".join(flags)
    cmd = f"TotalSegmentator -i {in_path} -o {out_dir} --roi_subset liver {flag_str}".strip()
    run(cmd, env=os.environ.copy())
    out_path = os.path.join(out_dir, "liver.nii.gz")
    if not os.path.exists(out_path):
        got = [p for p in os.listdir(out_dir) if p.endswith(".nii.gz")]
        raise RuntimeError(f"TotalSegmentator (liver): expected liver.nii.gz, found: {got}")
    return out_path

def totalseg_multilabel(in_path: str, out_dir: str, fast: bool = False) -> str:
    """
    Optional: keep multilabel endpoint, but normalize filenames.
    """
    flags = "--ml --fast" if fast else "--ml"
    cmd = f"TotalSegmentator -i {in_path} -o {out_dir} {flags}"
    run(cmd, env=os.environ.copy())
    # TS has used both 'segmentation.nii.gz' and 'segmentations.nii.gz' historically
    for name in ("segmentation.nii.gz", "segmentations.nii.gz"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    raise RuntimeError("TotalSegmentator (ml): multi-label file not found")

@app.get("/healthz")
def health():
    return PlainTextResponse("ok")

@app.get("/version")
def version():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import nnunet
        info["nnunet_v1"] = getattr(nnunet, "__version__", "unknown")
    except Exception as e:
        info["nnunet_v1_error"] = str(e)
    try:
        import totalsegmentator as ts
        info["totalseg"] = getattr(ts, "__version__", "unknown")
    except Exception as e:
        info["totalseg_error"] = str(e)
    return info

@app.post("/segment/task008")
async def segment_task008(ct: UploadFile = File(...), folds: str = "0"):
    cid = f"case_{uuid.uuid4().hex[:8]}"
    in_dir, out_dir = os.path.join(IN_ROOT, cid), os.path.join(OUT_ROOT, cid)
    os.makedirs(in_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)

    in_path = os.path.join(in_dir, f"{cid}_0000.nii.gz")
    with open(in_path, "wb") as f:
        f.write(await ct.read())

    t0 = time.time()
    try:
        out_path = nnunet_v1_task008(in_dir, out_dir, case_id=cid, folds=folds)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "Task008 failed", "detail": e.stderr or e.output})

    print(f"[task008] {cid} done in {time.time()-t0:.1f}s", flush=True)
    return FileResponse(out_path, media_type="application/gzip", filename=f"{cid}_task008.nii.gz")

@app.post("/segment/liver")
async def segment_liver(ct: UploadFile = File(...), fast: bool = False):
    """
    TotalSegmentator liver-only, predictable output liver.nii.gz.
    """
    cid = f"case_{uuid.uuid4().hex[:8]}"
    in_dir, out_dir = os.path.join(IN_ROOT, cid), os.path.join(OUT_ROOT, cid)
    os.makedirs(in_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    in_path = os.path.join(in_dir, f"{cid}.nii.gz")
    with open(in_path, "wb") as f: f.write(await ct.read())
    t0 = time.time()
    try:
        out_path = totalseg_liver_only(in_path, out_dir, fast=fast)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "TotalSegmentator liver failed", "detail": e.stderr or e.output})
    print(f"[liver] {cid} done in {time.time()-t0:.1f}s", flush=True)
    return FileResponse(out_path, media_type="application/gzip", filename=f"{cid}_liver.nii.gz")

@app.post("/segment/totalseg")
async def segment_totalseg(ct: UploadFile = File(...), fast: bool = False):
    """
    Optional: multi-label output for debugging/research.
    """
    cid = f"case_{uuid.uuid4().hex[:8]}"
    in_dir, out_dir = os.path.join(IN_ROOT, cid), os.path.join(OUT_ROOT, cid)
    os.makedirs(in_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    in_path = os.path.join(in_dir, f"{cid}.nii.gz")
    with open(in_path, "wb") as f: f.write(await ct.read())
    t0 = time.time()
    try:
        out_path = totalseg_multilabel(in_path, out_dir, fast=fast)
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "TotalSegmentator multi-label failed", "detail": e.stderr or e.output})
    print(f"[totalseg-ml] {cid} done in {time.time()-t0:.1f}s", flush=True)
    return FileResponse(out_path, media_type="application/gzip", filename=f"{cid}_totalseg.nii.gz")

@app.post("/segment/both")
async def segment_both(
    ct: UploadFile = File(...),
    folds: str = "0",
    fast: bool = True,
    background_tasks: BackgroundTasks = None,
):
    cid = f"case_{uuid.uuid4().hex[:8]}"
    case_root = os.path.join(OUT_ROOT, cid)
    out_ts  = os.path.join(case_root, "totalseg")
    out_t8  = os.path.join(case_root, "task008")
    case_in = os.path.join(IN_ROOT, cid)
    os.makedirs(case_in, exist_ok=True)
    os.makedirs(out_ts, exist_ok=True)
    os.makedirs(out_t8, exist_ok=True)

    # Write once; TS uses raw, nnUNet v1 needs _0000
    raw_ct = os.path.join(case_in, f"{cid}.nii.gz")
    ct_v1  = os.path.join(case_in, f"{cid}_0000.nii.gz")
    data = await ct.read()
    with open(raw_ct, "wb") as f: f.write(data)
    with open(ct_v1,  "wb") as f: f.write(data)

    timings = {}

    # TotalSegmentator liver-only
    t = time.time()
    liver_path = totalseg_liver_only(raw_ct, out_ts, fast=fast)  # -> <out_ts>/liver.nii.gz
    timings["liver_s"] = round(time.time() - t, 2)

    # Task008
    t = time.time()
    t8_path = nnunet_v1_task008(case_in, out_t8, case_id=cid, folds=folds)  # -> <out_t8>/<cid>.nii.gz
    timings["task008_s"] = round(time.time() - t, 2)

    # Normalize names into a "package" dir so the zip is simple
    pkg_dir = os.path.join(case_root, "package")
    os.makedirs(pkg_dir, exist_ok=True)

    final_liver = os.path.join(pkg_dir, "liver.nii.gz")
    final_task8 = os.path.join(pkg_dir, "task008.nii.gz")
    shutil.copy2(liver_path, final_liver)
    shutil.copy2(t8_path, final_task8)

    meta = {
        "case_id": cid,
        "labels_task008": {"1": "hepatic_vessels", "2": "liver_tumors"},
        "timestamp": time.time(),
        **timings,
    }
    meta_path = os.path.join(pkg_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    zip_path = shutil.make_archive(os.path.join("/tmp", f"{cid}_results"), "zip", pkg_dir)

    if background_tasks:
        background_tasks.add_task(shutil.rmtree, case_in, ignore_errors=True)
        background_tasks.add_task(shutil.rmtree, case_root, ignore_errors=True)
        background_tasks.add_task(os.remove, zip_path)

    return FileResponse(zip_path, media_type="application/zip", filename=f"{cid}_results.zip")
