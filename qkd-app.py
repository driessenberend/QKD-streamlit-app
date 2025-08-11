import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.visualization import circuit_drawer
    from qiskit.exceptions import MissingOptionalLibraryError
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Optional visualization deps
try:
    import matplotlib  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
try:
    import pylatexenc  # noqa: F401
    PYLATEXENC_AVAILABLE = True
except Exception:
    PYLATEXENC_AVAILABLE = False


def randbits(n, rng):
    return rng.integers(0, 2, size=n, dtype=np.int8)

def randbases(n, rng):
    return rng.integers(0, 2, size=n, dtype=np.int8)

def encode_measure_probabilistic(alice_bits, alice_bases, bob_bases, eve_enabled, noise, rng):
    n = len(alice_bits)
    effective_bits = alice_bits.copy()
    if eve_enabled:
        eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)
        eve_wrong_basis = eve_bases ^ alice_bases
        flips_from_eve = (eve_wrong_basis * rng.integers(0, 2, size=n, dtype=np.int8))
        effective_bits ^= flips_from_eve
    noise_flips = (rng.random(n) < noise).astype(np.int8)
    effective_bits ^= noise_flips
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

def run_bb84(n_qubits, eve, noise, seed):
    rng = np.random.default_rng(seed)
    alice_bits = randbits(n_qubits, rng)
    alice_bases = randbases(n_qubits, rng)
    bob_bases = randbases(n_qubits, rng)
    bob_raw, sift_mask, sifted_key, qber = encode_measure_probabilistic(alice_bits, alice_bases, bob_bases, eve, noise, rng)
    return BB84Run(n_qubits, alice_bits, alice_bases, bob_bases, bob_raw, sift_mask, sifted_key, qber, eve, noise)

def bb84_circuit_sample(alice_bits, alice_bases, bob_bases, sample_size=8):
    if not QISKIT_AVAILABLE:
        return None
    n = min(sample_size, len(alice_bits))
    qc = QuantumCircuit(n, n)
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 1:
            qc.h(i)
    for i in range(n):
        if bob_bases[i] == 1:
            qc.h(i)
        qc.measure(i, i)
    return qc

def run_aer(qc, shots=1024):
    if not QISKIT_AVAILABLE:
        return {}
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()

st.set_page_config(page_title="QKD: BB84 Simulator (Qiskit)", page_icon="🔑", layout="wide")

st.title("🔑 Quantum Key Distribution — BB84, QBER & Circuits")

with st.sidebar:
    n_qubits = st.slider("Aantal qubits", 10, 2000, 200, step=10)
    eve_enabled = st.toggle("Voeg Eve (afluisteren) toe", value=False)
    noise = st.slider("Kanaalruis (bitflip-probabiliteit)", 0.0, 0.2, 0.02, step=0.01)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("---")
    render_mode = st.selectbox(
        "Circuit weergave",
        ["Automatisch", "Matplotlib", "Tekst"],
        help="Kies 'Tekst' als Matplotlib/pylatexenc problemen geeft op Streamlit Cloud."
    )

result = run_bb84(n_qubits, eve_enabled, noise, seed)

tab1, tab2, tab3 = st.tabs(["1) BB84-simulator", "2) Prestatie & QBER", "3) Live Qiskit-circuits"])

with tab1:
    df = pd.DataFrame({
        "i": np.arange(result.n_qubits),
        "Alice bit": result.alice_bits,
        "Alice basis": np.where(result.alice_bases==0, "Z", "X"),
        "Bob basis": np.where(result.bob_bases==0, "Z", "X"),
        "Bob raw": result.bob_raw,
        "Basis match": np.where(result.sift_mask, "✓", "✗"),
    })
    st.dataframe(df.head(200))
    st.metric("Sifted key length", int(result.sift_mask.sum()))
    qber_pct = (np.nan if np.isnan(result.qber) else 100*result.qber)
    st.metric("QBER (sifted)", f"{qber_pct:.2f}%")

with tab2:
    st.write("Monte Carlo sweep kan hier toegevoegd worden.")

with tab3:
    if not QISKIT_AVAILABLE:
        st.warning("Qiskit niet geïnstalleerd.")
    else:
        qc = bb84_circuit_sample(result.alice_bits, result.alice_bases, result.bob_bases, sample_size=8)
        if qc:
            # Bepaal of we Matplotlib mogen gebruiken
            want_mpl = (render_mode == "Matplotlib") or (
                render_mode == "Automatisch" and MATPLOTLIB_AVAILABLE and PYLATEXENC_AVAILABLE
            )
            if want_mpl and MATPLOTLIB_AVAILABLE and PYLATEXENC_AVAILABLE:
                try:
                    fig = circuit_drawer(qc, output='mpl', fold=-1)
                    st.pyplot(fig)
                except MissingOptionalLibraryError as e:
                    st.warning(
                        "Matplotlib drawer niet beschikbaar: " + str(e) +
                        "\nVal terug op tekstweergave. Voeg 'matplotlib' en 'pylatexenc' toe aan requirements.txt of pin Python 3.11 met runtime.txt."
                    )
                    st.text(qc.draw(output='text'))
                except Exception as e:
                    st.error(f"Kon het circuit niet tekenen ({type(e).__name__}). Vallen terug op tekst.")
                    st.text(qc.draw(output='text'))
            else:
                if render_mode == "Matplotlib" and (not MATPLOTLIB_AVAILABLE or not PYLATEXENC_AVAILABLE):
                    st.info("Matplotlib/pylatexenc niet gevonden. Toon tekstdiagram i.p.v. figuur.")
                st.text(qc.draw(output='text'))
