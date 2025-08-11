import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List

# Optional/Qiskit imports guarded so local runs without Qiskit still render UI
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import circuit_drawer
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# ----------------------------
# Utility & domain functions
# ----------------------------

def randbits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.int8)


def randbases(n: int, rng: np.random.Generator) -> np.ndarray:
    # 0 = Z basis (|0>, |1>), 1 = X basis (|+>, |->)
    return rng.integers(0, 2, size=n, dtype=np.int8)


def encode_measure_probabilistic(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    eve_enabled: bool,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fast, *classical* BB84 Monte Carlo model.
    - If Eve intercept-resends: she chooses random bases and measures. This introduces a 25% QBER in the sifted key (expected, noise=0).
    - noise is a bit-flip probability applied on the quantum channel (simple model).

    Returns:
      bob_raw: Bob's raw measurement bits
      sift_mask: mask where Bob's basis == Alice's basis
      sifted_key: sifted key after basis reconciliation (Alice vs Bob)
      qber: quantum bit error rate on sifted key
    """
    n = len(alice_bits)
    assert n == len(alice_bases) == len(bob_bases)

    # Channel bit after Eve/noise
    effective_bits = alice_bits.copy()

    if eve_enabled:
        eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)
        # If Eve's basis != Alice's, Eve measures wrong basis => 50% chance flip vs original
        eve_wrong_basis = eve_bases ^ alice_bases
        flips_from_eve = (eve_wrong_basis * rng.integers(0, 2, size=n, dtype=np.int8))
        effective_bits = effective_bits ^ flips_from_eve
        # Eve resends in her basis; if Bob differs from Eve, he has 50% chance error relative to Eve
        # We don't need to model resend explicitly; the error emerges when Bob measures with his basis vs original after Eve.

    # Channel noise: bit flip with prob=noise regardless of basis
    noise_flips = (rng.random(n) < noise).astype(np.int8)
    effective_bits = effective_bits ^ noise_flips

    # Bob measures: if basis matches Alice, he gets effective_bits; otherwise random
    basis_match = (bob_bases == alice_bases).astype(np.int8)
    random_bits = rng.integers(0, 2, size=n, dtype=np.int8)
    bob_raw = basis_match * effective_bits + (1 - basis_match) * random_bits

    sift_mask = basis_match.astype(bool)
    sifted_alice = alice_bits[sift_mask]
    sifted_bob = bob_raw[sift_mask]

    if len(sifted_alice) == 0:
        return bob_raw, sift_mask, np.array([], dtype=np.int8), float("nan")

    qber = float(np.mean(sifted_alice ^ sifted_bob))
    return bob_raw, sift_mask, sifted_bob, qber


@dataclass
class BB84Run:
    n_qubits: int
    alice_bits: np.ndarray
    alice_bases: np.ndarray
    bob_bases: np.ndarray
    bob_raw: np.ndarray
    sift_mask: np.ndarray
    sifted_key: np.ndarray
    qber: float
    eve_enabled: bool
    noise: float


def run_bb84(n_qubits: int, eve: bool, noise: float, seed: int) -> BB84Run:
    rng = np.random.default_rng(seed)
    alice_bits = randbits(n_qubits, rng)
    alice_bases = randbases(n_qubits, rng)
    bob_bases = randbases(n_qubits, rng)

    bob_raw, sift_mask, sifted_key, qber = encode_measure_probabilistic(
        alice_bits, alice_bases, bob_bases, eve, noise, rng
    )

    return BB84Run(
        n_qubits=n_qubits,
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        bob_raw=bob_raw,
        sift_mask=sift_mask,
        sifted_key=sifted_key,
        qber=qber,
        eve_enabled=eve,
        noise=noise,
    )


# ----------------------------
# Qiskit circuit helpers (for Tab 5: live circuits)
# ----------------------------

def bb84_circuit_sample(
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    sample_size: int = 8,
):
    if not QISKIT_AVAILABLE:
        return None
    n = min(sample_size, len(alice_bits))
    qc = QuantumCircuit(n, n)
    # Prepare Alice's states
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 1:  # X-basis -> apply Hadamard to move to |+> / |->
            qc.h(i)
    # Bob's measurement bases
    for i in range(n):
        if bob_bases[i] == 1:  # measure in X basis -> rotate with H, then Z-measure
            qc.h(i)
        qc.measure(i, i)
    return qc


def run_aer(qc: 'QuantumCircuit', shots: int = 1024) -> Dict[str, int]:
    if not QISKIT_AVAILABLE:
        return {}
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="QKD: BB84 Simulator (Qiskit)",
    page_icon="🔑",
    layout="wide",
)

st.title("🔑 Quantum Key Distribution — BB84, QBER & Circuits")
st.caption(
    "Interactieve demo van BB84 met optionele eavesdropper (Eve), ruis, prestatieanalyse en live Qiskit-circuits."
)

with st.sidebar:
    st.header("Parameters")
    n_qubits = st.slider("Aantal qubits", 10, 2000, 200, step=10)
    eve_enabled = st.toggle("Voeg Eve (afluisteren) toe", value=False)
    noise = st.slider("Kanaalruis (bitflip-probabiliteit)", 0.0, 0.2, 0.02, step=0.01)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("---")
    st.caption("Tip: verhoog qubits voor stabielere statistieken. Zonder ruis is QBER ≈ 0% (geen Eve) of ≈ 25% (met Eve).")

result = run_bb84(n_qubits, eve_enabled, noise, seed)

# Tabs: 1) BB84 stap-voor-stap 2) Prestatie/QBER 3) Qiskit-circuit
tab1, tab2, tab3 = st.tabs([
    "1) BB84-simulator",
    "2) Prestatie & QBER",
    "3) Live Qiskit-circuits",
])

with tab1:
    st.subheader("Stap-voor-stap BB84")
    colA, colB = st.columns([2, 1])
    with colA:
        df = pd.DataFrame({
            "i": np.arange(result.n_qubits),
            "Alice bit": result.alice_bits,
            "Alice basis": np.where(result.alice_bases==0, "Z", "X"),
            "Bob basis": np.where(result.bob_bases==0, "Z", "X"),
            "Bob raw": result.bob_raw,
            "Basis match": np.where(result.sift_mask, "✓", "✗"),
        })
        st.dataframe(df.head(200), use_container_width=True)
    with colB:
        st.metric("Sifted key length", int(result.sift_mask.sum()))
        qber_pct = (np.nan if np.isnan(result.qber) else 100*result.qber)
        st.metric("QBER (sifted)", f"{qber_pct:.2f}%")
        if len(result.sifted_key) > 0:
            st.code(
                """Sleutel (eerste 64 bits):\n""" + ''.join(map(str, result.sifted_key[:64]))
            )
        if eve_enabled:
            st.info("Eve is actief: verwacht ~25% QBER (zonder ruis).")
        elif noise > 0:
            st.info("Ruis actief: QBER ≈ noise in sifted key (benadering).")

with tab2:
    st.subheader("Prestatie & QBER-onderzoek")
    st.caption("Monte Carlo sweep over aantal qubits en ruis. Resultaat: gemiddelde QBER en sleutel-lengte.")
    col1, col2 = st.columns(2)
    with col1:
        sweep_qubits = st.slider("Sweep: maximum qubits", 50, 3000, max(500, n_qubits), step=50)
        step = st.select_slider("Stapgrootte", options=[25, 50, 100, 200], value=100)
    with col2:
        reps = st.slider("Herhalingen per punt", 1, 20, 5)
        sweep_noise = st.slider("Vaste ruis voor sweep", 0.0, 0.2, float(noise), step=0.01)
        sweep_eve = st.toggle("Eve tijdens sweep", value=eve_enabled)

    rng = np.random.default_rng(seed+123)
    qubit_grid = list(range(50, sweep_qubits+1, step))
    rows = []
    for q in qubit_grid:
        qb_qber = []
        qb_len = []
        for _ in range(reps):
            r = run_bb84(q, sweep_eve, sweep_noise, int(rng.integers(0, 1_000_000)))
            if not np.isnan(r.qber):
                qb_qber.append(r.qber)
                qb_len.append(int(r.sift_mask.sum()))
        rows.append({"qubits": q, "mean_qber": np.mean(qb_qber) if qb_qber else np.nan, "mean_keylen": np.mean(qb_len) if qb_len else 0})
    sweep_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(sweep_df.set_index("qubits")["mean_qber"], use_container_width=True, height=300)
        st.caption("Gemiddelde QBER vs. aantal qubits")
    with c2:
        st.line_chart(sweep_df.set_index("qubits")["mean_keylen"], use_container_width=True, height=300)
        st.caption("Gemiddelde sleutel-lengte (na sifting) vs. aantal qubits")

    st.dataframe(sweep_df, use_container_width=True)

with tab3:
    st.subheader("Live Qiskit-circuits (sample van eerste qubits)")
    if not QISKIT_AVAILABLE:
        st.warning(
            "Qiskit niet gevonden. Voeg 'qiskit' en 'qiskit-aer' toe aan requirements.txt in je deployment om deze tab te activeren."
        )
    else:
        qc = bb84_circuit_sample(result.alice_bits, result.alice_bases, result.bob_bases, sample_size=8)
        if qc is None:
            st.error("Kon circuit niet bouwen.")
        else:
            st.write("**Circuitvoorbeeld (eerste 8 qubits):**")
            st.pyplot(circuit_drawer(qc, output='mpl', fold=-1))
            if st.button("Run op AerSimulator (1024 shots)"):
                counts = run_aer(qc, shots=1024)
                st.write("Resultaat (counts):", counts)
                st.caption("Let op: door willekeurige bases lijken de bitstrings uniform verdeeld.")

st.markdown("---")
st.markdown(
    """
**Uitleg in het kort**  
- **BB84**: Alice encodeert bits in Z/X-bases; Bob meet in willekeurige bases. Na vergelijking behouden ze alleen posities waar de basis overeenkomt (sifted key).  
- **QBER**: Foutpercentage in de sifted key. Zonder ruis of Eve ≈ 0%. Met Eve (intercept-resend) ≈ 25%.  
- **Ruis**: Hier gemodelleerd als bitflip-probabiliteit op het kanaal.  
- **Circuits**: De circuitweergave toont de H-poorten voor basiskeuze en metingen per qubit.  
"""
)
