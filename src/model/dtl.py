from enum import IntEnum
from typing import Any, NamedTuple, Self

import numpy as np
from numpy.typing import NDArray

from .base import Arrays, BaseModel, FloatArray, IntArray, Params

_FLOAT_TOLERANCE = 1e-9

# pengali untuk menggabungkan (node, value) menjadi satu key yang bisa di-searchsorted
_EDGE_KEY_MULTIPLIER = 1 << 32


class _NodeType(IntEnum):
    """Tipe node di node_type_."""

    LEAF = 0
    NUMERIC_SPLIT = 1  # split biner pada threshold
    CATEGORICAL_SPLIT = 2  # split multiway


class _Edge(NamedTuple):
    """Satu branch categorical, dari node dengan nilai kategori menuju child dengan proporsi training prob."""

    node: int
    value: int
    child: int
    prob: float


class _SplitEvaluation(NamedTuple):
    """Hasil evaluasi satu fitur. threshold None untuk split categorical."""

    gain: float
    branch_weights: FloatArray
    threshold: float | None


class _SplitCandidate(NamedTuple):
    """Kandidat split pada satu fitur. threshold None untuk split categorical."""

    gain: float
    gain_ratio: float
    feature_idx: int
    threshold: float | None


class _GrowingTree:
    """Tree yang sedang dibangun, disimpan sebagai list paralel dengan index = id node (0 adalah root)."""

    def __init__(self) -> None:
        self.feature: list[int] = []
        self.node_type: list[_NodeType] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.left_prob: list[float] = []
        self.class_weights: list[FloatArray] = []
        self.edges: list[_Edge] = []

    def add_leaf(self, class_weights: FloatArray) -> int:
        """Tambah leaf baru dengan class_weights, return id node."""
        self.feature.append(-1)
        self.node_type.append(_NodeType.LEAF)
        self.threshold.append(np.nan)
        self.left.append(-1)
        self.right.append(-1)
        self.left_prob.append(0.5)
        self.class_weights.append(class_weights)
        return len(self.node_type) - 1

    def make_numeric_split(
        self,
        node: int,
        feature_idx: int,
        threshold: float,
        left_child: int,
        right_child: int,
        left_prob: float,
    ) -> None:
        """Ubah leaf menjadi split biner pada feature_idx <= threshold."""
        self.feature[node] = feature_idx
        self.node_type[node] = _NodeType.NUMERIC_SPLIT
        self.threshold[node] = threshold
        self.left[node] = left_child
        self.right[node] = right_child
        self.left_prob[node] = left_prob

    def make_categorical_split(
        self, node: int, feature_idx: int, edges: list[_Edge]
    ) -> None:
        """Ubah leaf menjadi split multiway pada feature_idx, satu edge per nilai."""
        self.feature[node] = feature_idx
        self.node_type[node] = _NodeType.CATEGORICAL_SPLIT
        self.edges.extend(edges)


class _Tokens(NamedTuple):
    """Jejak prediksi, berisi sample, node tempat ia berada, dan bobot yang dibawanya."""

    samples: IntArray
    nodes: IntArray
    weights: FloatArray

    def select(self, mask: NDArray[np.bool_]) -> "_Tokens":
        """Ambil token yang mask-nya True."""
        return _Tokens(self.samples[mask], self.nodes[mask], self.weights[mask])

    @staticmethod
    def concatenate(parts: list["_Tokens"]) -> "_Tokens":
        """Gabungkan beberapa kelompok token menjadi satu."""
        return _Tokens(
            np.concatenate([part.samples for part in parts]),
            np.concatenate([part.nodes for part in parts]),
            np.concatenate([part.weights for part in parts]),
        )


def _entropy(class_weights: FloatArray) -> FloatArray:
    """Entropy (log2) sepanjang axis terakhir, dengan class_weights (..., n_classes)."""
    total = class_weights.sum(axis=-1, keepdims=True)
    proportion = np.divide(
        class_weights,
        total,
        out=np.zeros_like(class_weights, dtype=np.float64),
        where=total > 0,
    )
    log_proportion = np.log2(
        proportion, out=np.zeros_like(proportion), where=proportion > 0
    )
    return np.asarray(-(proportion * log_proportion).sum(axis=-1))


def _split_info(branch_weights: FloatArray, total_weight: float) -> float:
    """Split info C4.5. Instance dengan missing value dihitung sebagai satu branch tambahan."""
    missing_weight = total_weight - branch_weights.sum()
    if missing_weight > _FLOAT_TOLERANCE:
        branch_weights = np.append(branch_weights, missing_weight)

    proportion = branch_weights[branch_weights > 0] / total_weight
    return float(-(proportion * np.log2(proportion)).sum())


class _Builder:
    """Build tree recursively.

    Parameter:
        X: fitur training, NaN menandai missing value.
        y: label training yang sudah di-encode menjadi 0..n_classes-1.
        n_classes: jumlah kelas.
        is_categorical: mask bool per kolom X, True jika kolom categorical.
        max_depth: kedalaman maksimum tree, None berarti tanpa batas.
        min_samples_split: bobot minimum sebuah node agar boleh di-split.
        min_gain: gain minimum agar sebuah split diterima.

    Usage:
        builder = _Builder(X, y, n_classes, is_categorical, max_depth, min_samples_split, min_gain)
        builder.grow(np.arange(len(X)), np.ones(len(X)), 0)
        tree = builder.tree

    Notes:
        - Hasil build ada di builder.tree (_GrowingTree), dengan root di id node 0.
        - Branch categorical disimpan di tree.edges sebagai _Edge.
    """

    def __init__(
        self,
        X: FloatArray,
        y: IntArray,
        n_classes: int,
        is_categorical: NDArray[np.bool_],
        max_depth: int | None,
        min_samples_split: int,
        min_gain: float,
    ) -> None:
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.is_categorical = is_categorical
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.tree = _GrowingTree()

    def grow(self, sample_idx: IntArray, weights: FloatArray, depth: int) -> int:
        """Build subtree dari instance sample_idx berbobot weights, return id node.
        Stop jika tidak ada split yang berguna.
        """
        class_weights = np.bincount(
            self.y[sample_idx], weights=weights, minlength=self.n_classes
        )
        node = self.tree.add_leaf(class_weights)
        total_weight = class_weights.sum()

        # berhenti jika node murni, terlalu kecil, atau mencapai max_depth
        if (
            class_weights.max() >= total_weight - _FLOAT_TOLERANCE
            or total_weight < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        best_split = self._best_split(self.X[sample_idx], self.y[sample_idx], weights)
        if best_split is None:
            return node

        if best_split.threshold is None:
            self._split_categorical(
                node, sample_idx, weights, best_split.feature_idx, depth
            )
        else:
            self._split_numeric(
                node,
                sample_idx,
                weights,
                best_split.feature_idx,
                best_split.threshold,
                depth,
            )
        return node

    # Private methods

    def _split_numeric(
        self,
        node: int,
        sample_idx: IntArray,
        weights: FloatArray,
        feature_idx: int,
        threshold: float,
        depth: int,
    ) -> None:
        """Pecah node pada column <= threshold. Instance missing diteruskan ke kedua branch secara fractional."""
        column = self.X[sample_idx, feature_idx]
        is_known = ~np.isnan(column)
        goes_left = is_known & (column <= threshold)
        goes_right = is_known & (column > threshold)
        left_weight = weights[goes_left].sum()
        right_weight = weights[goes_right].sum()
        left_prob = float(left_weight / (left_weight + right_weight))

        # instance missing masuk ke kedua branch dengan bobot proporsional ukuran branch
        missing_idx, missing_weights = sample_idx[~is_known], weights[~is_known]
        left_child = self._grow_branch(
            sample_idx[goes_left],
            weights[goes_left],
            missing_idx,
            missing_weights,
            left_prob,
            depth,
        )
        right_child = self._grow_branch(
            sample_idx[goes_right],
            weights[goes_right],
            missing_idx,
            missing_weights,
            1 - left_prob,
            depth,
        )
        self.tree.make_numeric_split(
            node, feature_idx, float(threshold), left_child, right_child, left_prob
        )

    def _split_categorical(
        self,
        node: int,
        sample_idx: IntArray,
        weights: FloatArray,
        feature_idx: int,
        depth: int,
    ) -> None:
        """Pecah node menjadi satu branch per nilai. Instance missing diteruskan ke semua branch secara fractional."""
        column = self.X[sample_idx, feature_idx]
        is_known = ~np.isnan(column)

        # kelompokkan instance per nilai dengan sort, lalu split di batas nilai
        known_positions = np.flatnonzero(is_known)
        known_positions = known_positions[
            np.argsort(column[known_positions], kind="stable")
        ]
        sorted_values = column[known_positions]
        group_starts = np.flatnonzero(np.diff(sorted_values)) + 1
        groups = np.split(known_positions, group_starts)
        group_values = sorted_values[
            np.concatenate([[0], group_starts]).astype(np.int64)
        ]

        branch_weights = np.array([weights[group].sum() for group in groups])
        branch_probs = branch_weights / branch_weights.sum()
        missing_idx = sample_idx[~is_known]
        missing_weights = weights[~is_known]
        children = [
            self._grow_branch(
                sample_idx[group],
                weights[group],
                missing_idx,
                missing_weights,
                prob,
                depth,
            )
            for group, prob in zip(groups, branch_probs)
        ]
        edges = [
            _Edge(node, int(value), child, float(prob))
            for value, child, prob in zip(group_values, children, branch_probs)
        ]
        self.tree.make_categorical_split(node, feature_idx, edges)

    def _grow_branch(
        self,
        branch_idx: IntArray,
        branch_weights: FloatArray,
        missing_idx: IntArray,
        missing_weights: FloatArray,
        branch_prob: float,
        depth: int,
    ) -> int:
        """Build child dari instance branch ditambah instance missing berbobot branch_prob."""
        return self.grow(
            np.concatenate([branch_idx, missing_idx]),
            np.concatenate([branch_weights, missing_weights * branch_prob]),
            depth + 1,
        )

    def _best_split(
        self, X_node: FloatArray, y_node: IntArray, weights: FloatArray
    ) -> _SplitCandidate | None:
        """Return kandidat dengan gain ratio terbaik, atau None jika tidak ada split dengan gain positif."""
        total_weight = weights.sum()
        candidates = [
            candidate
            for feature_idx in range(X_node.shape[1])
            if (
                candidate := self._evaluate_feature(
                    X_node, y_node, weights, feature_idx, total_weight
                )
            )
            is not None
        ]
        if not candidates:
            return None

        # C4.5: ambil gain ratio tertinggi hanya di antara atribut dengan gain >= rata-rata
        mean_gain = float(np.mean([candidate.gain for candidate in candidates]))
        eligible = [
            candidate
            for candidate in candidates
            if candidate.gain >= mean_gain - _FLOAT_TOLERANCE
        ]
        return max(eligible, key=lambda candidate: candidate.gain_ratio)

    def _evaluate_feature(
        self,
        X_node: FloatArray,
        y_node: IntArray,
        weights: FloatArray,
        feature_idx: int,
        total_weight: float,
    ) -> _SplitCandidate | None:
        """Return kandidat split terbaik untuk satu fitur, atau None jika tidak ada yang layak."""
        column = X_node[:, feature_idx]
        is_known = ~np.isnan(column)
        if is_known.sum() < 2:
            return None

        known_column = column[is_known]
        known_y = y_node[is_known]
        known_weights = weights[is_known]
        known_class_weights = np.bincount(
            known_y, weights=known_weights, minlength=self.n_classes
        )

        if self.is_categorical[feature_idx]:
            evaluation = self._eval_categorical(
                known_column, known_y, known_weights, known_class_weights
            )
        else:
            evaluation = self._eval_numeric(
                known_column, known_y, known_weights, known_class_weights
            )
        if evaluation is None:
            return None

        # gain hanya dari instance yang nilainya diketahui, diskala dengan fraksinya
        gain = known_class_weights.sum() / total_weight * evaluation.gain
        if gain <= max(self.min_gain, _FLOAT_TOLERANCE):
            return None

        split_info = _split_info(evaluation.branch_weights, total_weight)
        if split_info <= _FLOAT_TOLERANCE:
            return None
        return _SplitCandidate(
            gain, gain / split_info, feature_idx, evaluation.threshold
        )

    def _eval_numeric(
        self,
        known_column: FloatArray,
        known_y: IntArray,
        known_weights: FloatArray,
        known_class_weights: FloatArray,
    ) -> _SplitEvaluation | None:
        """Cari threshold terbaik untuk kolom numeric, atau None jika semua nilai sama."""
        order = np.argsort(known_column, kind="stable")
        sorted_values = known_column[order]
        sorted_y = known_y[order]
        sorted_weights = known_weights[order]
        cut_positions = np.flatnonzero(sorted_values[1:] != sorted_values[:-1])
        if cut_positions.size == 0:
            return None

        # cumsum memberi left_class_weights untuk semua kandidat threshold sekaligus
        weight_per_class = np.zeros((len(sorted_values), self.n_classes))
        weight_per_class[np.arange(len(sorted_values)), sorted_y] = sorted_weights
        left_class_weights = np.cumsum(weight_per_class, axis=0)[cut_positions]
        right_class_weights = known_class_weights - left_class_weights
        left_weight = left_class_weights.sum(axis=1)
        right_weight = right_class_weights.sum(axis=1)
        split_entropy = (
            left_weight * _entropy(left_class_weights)
            + right_weight * _entropy(right_class_weights)
        ) / (left_weight + right_weight)
        best_cut = int(np.argmin(split_entropy))

        # penalti jumlah threshold kandidat (C4.5 Release 8, Quinlan 1996)
        gain = float(
            _entropy(known_class_weights)
            - split_entropy[best_cut]
            - np.log2(cut_positions.size) / known_class_weights.sum()
        )

        value_below_cut = sorted_values[cut_positions[best_cut]]
        value_above_cut = sorted_values[cut_positions[best_cut] + 1]
        threshold = (value_below_cut + value_above_cut) / 2
        # guard floating point supaya value_above_cut tetap di sisi kanan
        if threshold >= value_above_cut:
            threshold = value_below_cut

        branch_weights = np.array([left_weight[best_cut], right_weight[best_cut]])
        return _SplitEvaluation(gain, branch_weights, float(threshold))

    def _eval_categorical(
        self,
        known_column: FloatArray,
        known_y: IntArray,
        known_weights: FloatArray,
        known_class_weights: FloatArray,
    ) -> _SplitEvaluation | None:
        """Hitung gain split multiway untuk kolom categorical, atau None jika hanya ada satu nilai."""
        _, value_codes = np.unique(known_column, return_inverse=True)
        n_values = int(value_codes.max()) + 1
        if n_values < 2:
            return None

        weight_table = np.bincount(
            value_codes * self.n_classes + known_y,
            weights=known_weights,
            minlength=n_values * self.n_classes,
        ).reshape(n_values, self.n_classes)
        branch_weights = weight_table.sum(axis=1)
        split_entropy = float(
            (branch_weights * _entropy(weight_table)).sum() / branch_weights.sum()
        )
        gain = float(_entropy(known_class_weights)) - split_entropy
        return _SplitEvaluation(gain, branch_weights, None)


class C45DecisionTree(BaseModel):
    """Decision tree dengan algoritma C4.5.

    Parameter:
        max_depth: kedalaman maksimum tree, None berarti tanpa batas.
        min_samples_split: bobot minimum sebuah node agar boleh di-split.
        min_gain: gain minimum agar sebuah split diterima.
        categorical: index kolom X yang categorical. Nilainya harus integer code non-negatif.
        ccp_alpha: kekuatan cost-complexity pruning, 0 berarti tanpa pruning.

    Usage:
        model = C45DecisionTree(max_depth=5, categorical=[0, 3], ccp_alpha=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        model.save("model.npz")
        model = C45DecisionTree.load("model.npz")

    Notes:
        - NaN pada X menandai missing value, baik saat fit maupun predict.
        - Saat fit, instance missing dipecah secara fractional ke semua branch.
        - Saat predict, instance missing (atau kategori yang tidak pernah terlihat)
          mengikuti semua branch, probabilitasnya dijumlahkan dengan bobot proporsi training.
    """

    # Tree disimpan sebagai array dengan index = id node (0 adalah root).
    # edge_* menyimpan branch categorical: (edge_node, edge_value) -> edge_child.
    feature_: IntArray
    node_type_: IntArray
    threshold_: FloatArray
    left_: IntArray
    right_: IntArray
    left_prob_: FloatArray
    class_weights_: FloatArray
    class_distribution_: FloatArray
    edge_node_: IntArray
    edge_value_: IntArray
    edge_child_: IntArray
    edge_prob_: FloatArray
    edge_key_: IntArray
    edge_ptr_: IntArray
    classes_: NDArray[Any]
    n_features_: int

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_gain: float = 0.0,
        categorical: list[int] | None = None,
        ccp_alpha: float = 0.0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.categorical = categorical or []
        self.ccp_alpha = ccp_alpha

    def fit(self, X: FloatArray, y: IntArray) -> Self:
        """Latih tree pada X (NaN = missing value) dan y, lalu prune jika ccp_alpha > 0."""
        X = np.asarray(X, dtype=np.float64)
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        is_categorical = np.zeros(X.shape[1], dtype=bool)
        is_categorical[self.categorical] = True
        builder = _Builder(
            X,
            y_encoded,
            len(self.classes_),
            is_categorical,
            self.max_depth,
            self.min_samples_split,
            self.min_gain,
        )
        builder.grow(np.arange(len(X)), np.ones(len(X)), 0)

        self.n_features_ = X.shape[1]
        self._load_tree(builder.tree)
        self._sort_edges()
        if self.ccp_alpha > 0:
            self._prune()
        self._compact()
        self._index()
        return self

    def predict_proba(self, X: FloatArray) -> FloatArray:
        """Return probabilitas kelas. Branch missing dijumlahkan dengan bobot proporsi training."""
        # Setiap _Tokens adalah sekumpulan jejak (sample, node, bobot) yang turun satu level per iterasi.
        # Token missing bercabang ke semua child sehingga seluruh sample diproses vectorized.
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        proba = np.zeros((n_samples, len(self.classes_)))
        tokens = _Tokens(
            np.arange(n_samples),
            np.zeros(n_samples, dtype=np.int64),
            np.ones(n_samples),
        )

        while tokens.samples.size:
            token_types = self.node_type_[tokens.nodes]
            at_leaf = token_types == _NodeType.LEAF
            self._add_leaf_proba(proba, tokens.select(at_leaf))

            tokens = tokens.select(~at_leaf)
            token_types = token_types[~at_leaf]
            next_tokens: list[_Tokens] = []
            at_numeric = token_types == _NodeType.NUMERIC_SPLIT
            if at_numeric.any():
                next_tokens += self._advance_numeric(X, tokens.select(at_numeric))
            at_categorical = token_types == _NodeType.CATEGORICAL_SPLIT
            if at_categorical.any():
                next_tokens += self._advance_categorical(
                    X, tokens.select(at_categorical)
                )

            if not next_tokens:
                break
            tokens = _Tokens.concatenate(next_tokens)
        return proba

    def predict(self, X: FloatArray) -> IntArray:
        """Return label kelas dengan probabilitas tertinggi."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_state(self) -> tuple[Params, Arrays]:
        """Return (params, arrays) untuk save. Tree disimpan sebagai array."""
        params: Params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_gain": self.min_gain,
            "categorical": self.categorical,
            "ccp_alpha": self.ccp_alpha,
            "n_features": self.n_features_,
        }
        arrays: Arrays = {
            "classes": self.classes_,
            "feature": self.feature_,
            "node_type": self.node_type_,
            "threshold": self.threshold_,
            "left": self.left_,
            "right": self.right_,
            "left_prob": self.left_prob_,
            "class_weights": self.class_weights_,
            "edge_node": self.edge_node_,
            "edge_value": self.edge_value_,
            "edge_child": self.edge_child_,
            "edge_prob": self.edge_prob_,
        }
        return params, arrays

    def set_state(self, params: Params, arrays: Arrays) -> None:
        """Rebuild model dari hasil get_state."""
        self.max_depth = params["max_depth"]
        self.min_samples_split = params["min_samples_split"]
        self.min_gain = params["min_gain"]
        self.categorical = params["categorical"]
        self.ccp_alpha = params["ccp_alpha"]
        self.n_features_ = params["n_features"]

        self.classes_ = arrays["classes"]
        self.feature_ = arrays["feature"]
        self.node_type_ = arrays["node_type"]
        self.threshold_ = arrays["threshold"]
        self.left_ = arrays["left"]
        self.right_ = arrays["right"]
        self.left_prob_ = arrays["left_prob"]
        self.class_weights_ = arrays["class_weights"]
        self.edge_node_ = arrays["edge_node"]
        self.edge_value_ = arrays["edge_value"]
        self.edge_child_ = arrays["edge_child"]
        self.edge_prob_ = arrays["edge_prob"]
        self._index()

    # Private methods

    def _load_tree(self, tree: _GrowingTree) -> None:
        """Salin tree hasil build ke array atribut model."""
        self.feature_ = np.array(tree.feature, dtype=np.int64)
        self.node_type_ = np.array(tree.node_type, dtype=np.int64)
        self.threshold_ = np.array(tree.threshold, dtype=np.float64)
        self.left_ = np.array(tree.left, dtype=np.int64)
        self.right_ = np.array(tree.right, dtype=np.int64)
        self.left_prob_ = np.array(tree.left_prob, dtype=np.float64)
        self.class_weights_ = np.array(tree.class_weights, dtype=np.float64)
        self.edge_node_ = np.array([edge.node for edge in tree.edges], dtype=np.int64)
        self.edge_value_ = np.array([edge.value for edge in tree.edges], dtype=np.int64)
        self.edge_child_ = np.array([edge.child for edge in tree.edges], dtype=np.int64)
        self.edge_prob_ = np.array([edge.prob for edge in tree.edges], dtype=np.float64)

    def _sort_edges(self) -> None:
        """Urutkan edge berdasarkan (node, value) agar bisa dicari dengan searchsorted."""
        order = np.lexsort((self.edge_value_, self.edge_node_))
        self.edge_node_ = self.edge_node_[order]
        self.edge_value_ = self.edge_value_[order]
        self.edge_child_ = self.edge_child_[order]
        self.edge_prob_ = self.edge_prob_[order]

    def _children(self) -> list[list[int]]:
        """Return daftar child untuk tiap node (kosong untuk leaf)."""
        children: list[list[int]] = [[] for _ in range(len(self.node_type_))]
        for node in np.flatnonzero(self.node_type_ == _NodeType.NUMERIC_SPLIT):
            children[node] = [int(self.left_[node]), int(self.right_[node])]
        for node, child in zip(self.edge_node_, self.edge_child_):
            if self.node_type_[node] == _NodeType.CATEGORICAL_SPLIT:
                children[node].append(int(child))
        return children

    def _parents(self, children: list[list[int]]) -> IntArray:
        """Return parent tiap node (-1 untuk root)."""
        parent = np.full(len(children), -1, dtype=np.int64)
        for node, node_children in enumerate(children):
            for child in node_children:
                parent[child] = node
        return parent

    @staticmethod
    def _alpha(node_cost: float, subtree_cost: float, n_leaves: float) -> float:
        """Kenaikan cost per leaf yang dihemat jika subtree dijadikan satu leaf."""
        return (node_cost - subtree_cost) / (n_leaves - 1)

    def _subtree_stats(
        self, children: list[list[int]]
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Hitung (node_cost, subtree_cost, n_leaves, alpha) tiap node, bottom-up."""
        n_nodes = len(children)
        node_weight = self.class_weights_.sum(axis=1)
        node_cost = _entropy(self.class_weights_) * node_weight / node_weight[0]
        subtree_cost = node_cost.copy()
        n_leaves = np.ones(n_nodes)
        alpha = np.full(n_nodes, np.inf)

        # id child selalu lebih besar dari parent, jadi iterasi mundur berarti bottom-up
        for node in range(n_nodes - 1, -1, -1):
            if children[node]:
                subtree_cost[node] = sum(subtree_cost[c] for c in children[node])
                n_leaves[node] = sum(n_leaves[c] for c in children[node])
                alpha[node] = self._alpha(
                    node_cost[node], subtree_cost[node], n_leaves[node]
                )
        return node_cost, subtree_cost, n_leaves, alpha

    def _prune(self) -> None:
        """Cost-complexity pruning. Ubah node menjadi leaf selama alpha terkecil <= ccp_alpha.

        alpha(t) = (R(t) - R(subtree t)) / (jumlah leaf subtree - 1), dengan
        R = entropy * bobot node / bobot total.
        """
        children = self._children()
        parent = self._parents(children)
        node_cost, subtree_cost, n_leaves, alpha = self._subtree_stats(children)

        while True:
            weakest = int(np.argmin(alpha))
            if alpha[weakest] > self.ccp_alpha:
                break

            stack = list(children[weakest])
            while stack:
                descendant = stack.pop()
                alpha[descendant] = np.inf
                stack.extend(children[descendant])

            # prune node weakest, lalu perbarui cost dan jumlah leaf di semua ancestor
            self.node_type_[weakest] = _NodeType.LEAF
            cost_drop = subtree_cost[weakest] - node_cost[weakest]
            leaves_drop = n_leaves[weakest] - 1
            subtree_cost[weakest] = node_cost[weakest]
            n_leaves[weakest] = 1
            alpha[weakest] = np.inf

            ancestor = int(parent[weakest])
            while ancestor >= 0:
                subtree_cost[ancestor] -= cost_drop
                n_leaves[ancestor] -= leaves_drop
                alpha[ancestor] = self._alpha(
                    node_cost[ancestor], subtree_cost[ancestor], n_leaves[ancestor]
                )
                ancestor = int(parent[ancestor])

    def _compact(self) -> None:
        """Buang node yang tidak terjangkau dan renumber."""
        children = self._children()
        reachable = [0]
        for node in reachable:
            reachable.extend(children[node])
        old_ids = np.array(reachable, dtype=np.int64)
        new_id = np.full(len(self.node_type_), -1, dtype=np.int64)
        new_id[old_ids] = np.arange(len(old_ids))

        keep_edge = (new_id[self.edge_node_] >= 0) & (
            self.node_type_[self.edge_node_] == _NodeType.CATEGORICAL_SPLIT
        )
        self.edge_node_ = new_id[self.edge_node_[keep_edge]]
        self.edge_value_ = self.edge_value_[keep_edge]
        self.edge_child_ = new_id[self.edge_child_[keep_edge]]
        self.edge_prob_ = self.edge_prob_[keep_edge]

        is_numeric = self.node_type_[old_ids] == _NodeType.NUMERIC_SPLIT
        is_leaf = self.node_type_[old_ids] == _NodeType.LEAF
        self.left_ = np.where(is_numeric, new_id[self.left_[old_ids]], -1)
        self.right_ = np.where(is_numeric, new_id[self.right_[old_ids]], -1)
        self.feature_ = np.where(is_leaf, -1, self.feature_[old_ids])
        self.threshold_ = self.threshold_[old_ids]
        self.left_prob_ = self.left_prob_[old_ids]
        self.class_weights_ = self.class_weights_[old_ids]
        self.node_type_ = self.node_type_[old_ids]
        self._sort_edges()

    def _index(self) -> None:
        """Hitung struktur turunan, yaitu class_distribution_, edge_key_, dan edge_ptr_."""
        # edge_ptr_[i]:edge_ptr_[i+1] adalah rentang edge milik node i
        self.class_distribution_ = self.class_weights_ / self.class_weights_.sum(
            axis=1, keepdims=True
        )
        self.edge_key_ = self.edge_node_ * _EDGE_KEY_MULTIPLIER + self.edge_value_
        self.edge_ptr_ = np.searchsorted(
            self.edge_node_, np.arange(len(self.node_type_) + 1)
        )

    def _add_leaf_proba(self, proba: FloatArray, tokens: _Tokens) -> None:
        """Tambahkan distribusi kelas leaf, berbobot token, ke proba tiap sample."""
        for class_idx in range(proba.shape[1]):
            proba[:, class_idx] += np.bincount(
                tokens.samples,
                weights=tokens.weights
                * self.class_distribution_[tokens.nodes, class_idx],
                minlength=len(proba),
            )

    def _advance_numeric(self, X: FloatArray, tokens: _Tokens) -> list[_Tokens]:
        """Turunkan token di node numeric. Token missing bercabang ke left dan right sesuai left_prob_."""
        values = X[tokens.samples, self.feature_[tokens.nodes]]
        is_missing = np.isnan(values)
        known = tokens.select(~is_missing)
        missing = tokens.select(is_missing)

        known_child = np.where(
            values[~is_missing] <= self.threshold_[known.nodes],
            self.left_[known.nodes],
            self.right_[known.nodes],
        )
        left_prob = self.left_prob_[missing.nodes]
        return [
            _Tokens(known.samples, known_child, known.weights),
            _Tokens(
                missing.samples,
                self.left_[missing.nodes],
                missing.weights * left_prob,
            ),
            _Tokens(
                missing.samples,
                self.right_[missing.nodes],
                missing.weights * (1 - left_prob),
            ),
        ]

    def _advance_categorical(self, X: FloatArray, tokens: _Tokens) -> list[_Tokens]:
        """Turunkan token di node categorical. Kategori yang tidak punya edge bercabang ke semua child."""
        values = X[tokens.samples, self.feature_[tokens.nodes]]

        # kategori yang tidak ada di edge (termasuk NaN) diperlakukan sebagai missing
        is_usable = ~np.isnan(values) & (values >= 0)
        keys = tokens.nodes * _EDGE_KEY_MULTIPLIER + np.where(
            is_usable, values, 0
        ).astype(np.int64)
        edge_position = np.minimum(
            np.searchsorted(self.edge_key_, keys), len(self.edge_key_) - 1
        )
        is_found = is_usable & (self.edge_key_[edge_position] == keys)

        found = tokens.select(is_found)
        return [
            _Tokens(
                found.samples,
                self.edge_child_[edge_position[is_found]],
                found.weights,
            ),
            self._fan_out(tokens.select(~is_found)),
        ]

    def _fan_out(self, tokens: _Tokens) -> _Tokens:
        """Ganti tiap token dengan satu token per edge milik node-nya, berbobot edge_prob_."""
        first_edge = self.edge_ptr_[tokens.nodes]
        n_edges = self.edge_ptr_[tokens.nodes + 1] - first_edge
        source_token = np.repeat(np.arange(len(tokens.samples)), n_edges)
        offset_in_node = np.arange(n_edges.sum()) - np.repeat(
            np.cumsum(n_edges) - n_edges, n_edges
        )
        edge_idx = first_edge[source_token] + offset_in_node
        return _Tokens(
            tokens.samples[source_token],
            self.edge_child_[edge_idx],
            tokens.weights[source_token] * self.edge_prob_[edge_idx],
        )
