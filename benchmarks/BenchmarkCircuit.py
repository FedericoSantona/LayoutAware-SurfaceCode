# benchmark/BenchmarkCircuit.py
from __future__ import annotations
# Ensure Qiskit is available; raise immediately if not.
try:
    from qiskit.circuit import QuantumCircuit  # noqa: F401
except Exception as e:
    raise ImportError("Qiskit is required for metrics: install with 'pip install qiskit'.") from e

try:
    import yaml  # type: ignore
except Exception as e:
      raise ImportError("PyYAML is required for YAML export: install with 'pip install pyyaml'.") from e

from abc import ABC, abstractmethod


"""
BenchmarkCircuit: minimal, strict base class for logical benchmarks.

Design goals:
  • Empty __init__ — subclasses must implement build_circuit() which returns a Qiskit QuantumCircuit.
  • Stable logical metrics:
      - 'n_qubits'  : number of logical qubits in the abstract circuit.
      - 'depth'     : number of parallel logical layers (includes measurement & reset).
      - 'twoq'      : count of logical two‑qubit gates BEFORE any routing/decomposition.
  • Simple accessors and serializers:
      - get_circuit()  -> return the stored Qiskit QuantumCircuit.
      - to_qasm()      -> return OpenQASM string.
      - to_yaml()      -> return a YAML string description.

Strictness policy:
  • Qiskit and PyYAML must be installed; methods raise ImportError if not.
  • No hidden decompositions: SWAP counts as ONE logical 2Q op here (pre‑mapping).
"""

from typing import Any, Dict, List, Optional, Union


# --- Helpers -----------------------------------------------------------------
def _param_to_serializable(p: Any) -> Union[float, int, str]:
  """
  Convert a gate parameter (which might be a Qiskit ParameterExpression) into a
  JSON/YAML‑friendly primitive. We attempt numeric conversion first, then fall
  back to string without any clever heuristics (strict, predictable behavior).
  """
  try:
    v = float(p)                  # works for numeric literals & bound ParameterExpressions
    return int(v) if v.is_integer() else v
  except Exception:
    return str(p)                 # unbound symbols become explicit strings


# --- Core class --------------------------------------------------------------
class BenchmarkCircuit(ABC):
  """
  Abstract base class; daughter classes build the circuit by overriding build_circuit().

  Typical usage in a subclass:
      class Bell(BenchmarkCircuit):
          def build_circuit(self):
              from qiskit import QuantumCircuit
              qc = QuantumCircuit(2, 2)
              qc.h(0); qc.cx(0, 1); qc.measure([0,1],[0,1])
              return qc
  """

  def __init__(self) -> None:
    """
    Empty initializer. Subclasses MUST implement build_circuit() which returns
    a Qiskit QuantumCircuit. The base class lazily builds and caches it when
    first accessed via get_circuit().
    """
    self._qc = None  # type: Optional["QuantumCircuit"]

  # ---- Subclass hook --------------------------------------------------------
  @abstractmethod
  def build_circuit(self) -> "QuantumCircuit":
    """
    Construct and return the logical (pre-mapping) Qiskit QuantumCircuit.
    Subclasses must implement this method.
    """
    raise NotImplementedError

  # ---- Accessors ------------------------------------------------------------
  def get_circuit(self) -> "QuantumCircuit":
    """
    Return the stored Qiskit QuantumCircuit.
    Raises:
      - ValueError if no circuit has been set by the subclass yet.
    """
    if self._qc is None:
        qc = self.build_circuit()
        if not isinstance(qc, QuantumCircuit):
            raise TypeError("build_circuit() must return a qiskit.circuit.QuantumCircuit.")
        self._qc = qc
    return self._qc

  # ---- Logical metrics (pre‑mapping) ---------------------------------------
  def compute_logical_metrics(self) -> Dict[str, int]:
    """
    Extract the three logical metrics on the abstract circuit (pre-mapping).

    Counting rules (strict and recipe-aligned):
      • Logical qubit count:     qc.num_qubits.
      • Depth:                   qc.depth(), which includes measurement & reset and respects barriers.
      • Two-qubit gate count:    count native 2Q ops present in the logical circuit
                                 WITHOUT decomposing or adding routing overhead.
                                 Notably, 'swap' counts as ONE two-qubit op here.

    Returns:
      dict with keys {'n_qubits', 'depth', 'twoq'}.
    """
    qc = self.get_circuit()
 
    # 1) Logical qubit count
    n_qubits = int(qc.num_qubits)

    # 2) Depth including measurement/reset (Qiskit counts all ops; barriers affect layering)
    depth = int(qc.depth())

    # 3) Two‑qubit gate count at the logical level (no decomposition)
    ops = dict(qc.count_ops())
    twoq_gates = {
      # Control‑based & exchange interactions commonly seen at logical level
      "cx", "cz", "cp", "csx", "ecr", "iswap",
      # Parametric two‑qubit rotations
      "rxx", "ryy", "rzz", "xx_plus_yy",
    }
    twoq = sum(int(ops.get(g, 0)) for g in twoq_gates) + int(ops.get("swap", 0))

    return {"n_qubits": n_qubits, "depth": depth, "twoq": twoq}

  # ---- Serialization: QASM & YAML ------------------------------------------
  def to_qasm(self) -> str:
    """
    Return the circuit as an OpenQASM string.
    Strict: relies on Qiskit; if it cannot produce QASM, raise an error.
    """
    qc = self.get_circuit()
    try:
      if hasattr(qc, "qasm"):
        return qc.qasm()  # Qiskit < 1.0 still provides QuantumCircuit.qasm()

      # Qiskit >= 1.0 removed QuantumCircuit.qasm(); use qasm2 helper instead.
      try:
        from qiskit import qasm2  # type: ignore
      except Exception as e:  # pragma: no cover - surfaced as RuntimeError below
        raise RuntimeError("qiskit.qasm2 is required to export OpenQASM 2.0 in this environment.") from e

      return qasm2.dumps(qc)
    except Exception as e:
      raise RuntimeError(f"Failed to produce QASM from the circuit: {e}") from e

  def to_yaml(self) -> str:
    """
    Return a YAML string with a compact, explicit description of the circuit.

    Schema:
      version: 1
      name: <qc.name>
      qubits: <int>
      clbits: <int>
      instructions:
        - name: <op>
          qubits: [i, j, ...]
          clbits: [k, ...]             # only if present
          params:  [ ... ]             # numeric/strings, only if present
          condition: {on: REG, value: v}  # only if present

    Strict: requires PyYAML; no fallbacks or custom dumpers.
    """
    qc = self.get_circuit()

    # Require PyYAML explicitly; if absent, raise.
    
    # Build a YAML‑friendly payload using integer indices for qubits/clbits.
    q_index = {q: i for i, q in enumerate(qc.qubits)}
    c_index = {c: i for i, c in enumerate(qc.clbits)}

    instrs: List[Dict[str, Any]] = []
    for ci in qc.data:  # Qiskit 2.x: each item is a CircuitInstruction
      instr = ci.operation
      qargs = ci.qubits
      cargs = ci.clbits
      rec: Dict[str, Any] = {"name": instr.name}
      if qargs:
        rec["qubits"] = [int(q_index[q]) for q in qargs]
      if cargs:
        rec["clbits"] = [int(c_index[c]) for c in cargs]
      if getattr(instr, "params", None):
        params = [_param_to_serializable(p) for p in instr.params]
        if params:
          rec["params"] = params
      # Qiskit 2.x: classical condition lives on the CircuitInstruction
      cond = getattr(ci, "condition", None)
      if cond is not None:
        reg, val = cond  # tuple[Clbit or ClassicalRegister, int]
        rec["condition"] = {"on": getattr(reg, "name", str(reg)), "value": int(val)}
      instrs.append(rec)

    payload: Dict[str, Any] = {
      "version": 1,
      "name": getattr(qc, "name", "circuit"),
      "qubits": int(qc.num_qubits),
      "clbits": int(qc.num_clbits),
      "instructions": instrs,
    }

    # Serialize strictly via PyYAML; sort_keys=False preserves human‑readable order.
    return yaml.safe_dump(payload, sort_keys=False)
