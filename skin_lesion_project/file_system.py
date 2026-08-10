"""Read and write files on S3 (s3://bucket/key) or local disk."""
import io
import os
import pickle
import tempfile

IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")
_client = None


def s3():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3")
    return _client


def read_bytes(path):
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        return s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path, data):
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        s3().put_object(Bucket=bucket, Key=key, Body=data)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def read(path, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    data = read_bytes(path)
    buf = io.BytesIO(data)

    if ext in IMAGES:
        from PIL import Image
        img = Image.open(buf)
        img.load()
        return img
    if ext in (".pkl", ".pickle"):
        return pickle.loads(data)
    if ext == ".csv":
        import pandas as pd
        return pd.read_csv(buf, **kwargs)
    if ext in (".xlsx", ".xls", ".xlsm"):
        import pandas as pd
        return pd.read_excel(buf, **kwargs)
    if ext in (".pth", ".pt"):
        import torch
        return torch.load(buf, map_location=kwargs.pop("map_location", "cpu"), **kwargs)
    if ext in (".keras", ".h5"):
        from tensorflow import keras
        with tempfile.TemporaryDirectory() as d:
            local = os.path.join(d, os.path.basename(path))
            with open(local, "wb") as f:
                f.write(data)
            return keras.models.load_model(local, **kwargs)
    return data


def write(obj, path, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    buf = io.BytesIO()

    if ext in IMAGES:
        from PIL import Image
        if not isinstance(obj, Image.Image):
            import numpy as np
            obj = Image.fromarray(np.asarray(obj).astype("uint8").squeeze())
        fmt = "JPEG" if ext in (".jpg", ".jpeg") else ext[1:].upper()
        if fmt == "JPEG" and obj.mode in ("RGBA", "P", "LA"):
            obj = obj.convert("RGB")
        obj.save(buf, format=fmt, **kwargs)
    elif ext in (".pkl", ".pickle"):
        buf.write(pickle.dumps(obj))
    elif ext == ".csv":
        buf.write(obj.to_csv(index=kwargs.pop("index", False), **kwargs).encode())
    elif ext in (".xlsx", ".xls", ".xlsm"):
        import pandas as pd
        with pd.ExcelWriter(buf) as writer:
            obj.to_excel(writer, index=kwargs.pop("index", False), **kwargs)
    elif ext in (".pth", ".pt"):
        import torch
        if hasattr(obj, "state_dict") and not isinstance(obj, dict):
            obj = obj.state_dict()
        torch.save(obj, buf, **kwargs)
    elif ext in (".keras", ".h5"):
        with tempfile.TemporaryDirectory() as d:
            local = os.path.join(d, os.path.basename(path))
            obj.save(local, **kwargs)
            with open(local, "rb") as f:
                buf.write(f.read())
    else:
        buf.write(obj if isinstance(obj, bytes) else str(obj).encode())

    write_bytes(path, buf.getvalue())
