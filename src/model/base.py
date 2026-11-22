import json
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
Params = dict[str, Any]
Arrays = dict[str, NDArray[Any]]


class BaseModel:
    """Subclass implement fit/predict dan get_state/set_state.
    Persistence: <path>.json menyimpan scalar dan struktur, <path>.npz menyimpan array.
    """

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        raise NotImplementedError

    def predict(self, X: FloatArray) -> IntArray:
        raise NotImplementedError

    def predict_proba(self, X: FloatArray) -> FloatArray:
        raise NotImplementedError

    def get_state(self) -> tuple[Params, Arrays]:
        """Return (params: dict yang JSON-serializable, arrays: dict[str, np.ndarray])."""
        raise NotImplementedError

    def set_state(self, params: Params, arrays: Arrays) -> None:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params, arrays = self.get_state()
        meta = {"class": type(self).__name__, "params": params}
        path.with_suffix(".json").write_text(json.dumps(meta))
        np.savez_compressed(path.with_suffix(".npz"), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["class"] != cls.__name__:
            raise ValueError(f"Saved model is {meta['class']}, not {cls.__name__}")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        obj = cls.__new__(cls)
        obj.set_state(meta["params"], arrays)
        return obj
