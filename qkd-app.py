import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# ----------------------------
# App setup
# ----------------------------
st.set_page_config(
    page_title="QKD: BB84 Simulator (Qiskit)",
    page_icon="🔑",
    layout="wide",
)

st.title("🔑 Quantum Key Distribution — BB84, QBER & Circuits")
st.caption(
    "Interactieve demo van BB84 met optionele eavesdropper (Eve), ruis, prestatieanalyse en live Qiskit-circuits. "
    "De app start snel dankzij lazy imports en fallbacks."
)

# ----------------------------
# BB84 domain logic (puur NumPy, geen Qiskit nodig)
# ----------------------------
def randbits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.int8)

def randbases(n: int, rng: np.random.Generator) -> np.ndarray:
    # 0 = Z basis, 1 = X basis
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
    Snelle *klassieke* BB84 Monte Carlo model.
    - Eve (intercept-resend) introduceert ~25% QBER (zonder ruis) in de sifted key.
    - noise = bitflip-probabiliteit op het kanaal.

    Returns:
      bob_raw, sift_mask, sifted_key, qber
    """
    n = len(alice_bits)
    assert n == len(alice_bases) == len(bob_bases)

    effective_bits = alice_bits.copy()

    if eve_enabled:
        eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)
        eve_wrong_basis = eve_bases ^ alice_bases  # 1 als fout
        flips_from_eve = (eve_wrong_basis * rng.integers(0, 2, size=n, dtype=np.int8))
        effective_bits = effective_bits ^ flips_from_eve

    # Kanaalruis: bitflip met kans=noise
    noise_flips = (rng.random(n) < noise).astype(np.int8)
    effective_bits = effective_bits ^ noise_flips

    # Bob meet: match -> krijgt effective_bits, anders random bit
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
# Lazy Qiskit loader (alle zware imports pas op verzoek)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_qiskit():
    """
    Laadt Qiskit en checkt optionele teken-deps.
    Returns:
      dict met keys:
        ok (bool), err (Exception|None),
        QuantumCircuit, circuit_drawer, MissingOptionalLibraryError,
        AerSimulator (of None),
        have_matplotlib (bool), have_pylatexenc (bool)
    """
    try:
        from qiskit import QuantumCircuit, transpile
        try:
            from qiskit_aer import AerSimulator
        except Exception:
            AerSimulator = None  # simulator optioneel
        try:
            from qiskit.visualization import circuit_drawer
        except Exception as e:
            # circuit_drawer zelf kan ontbreken in exotische builds
            circuit_drawer = None

        try:
            from qiskit.exceptions import MissingOptionalLibraryError
        except Exception:
            class MissingOptionalLibraryError(Exception):
                pass

        # Optionele teken-deps detecteren
        try:
            import matplotlib  # noqa: F401
            have_matplotlib = True
        except Exception:
            have_matplotlib = False
        try:
            import pylatexenc  # noqa: F401
            have_pylatexenc = True
        except Exception:
            have_pylatexenc = False

        return dict(
            ok=True, err=None,
            QuantumCircuit=QuantumCircuit,
            transpile=transpile,
            circuit_drawer=circuit_drawer,
            MissingOptionalLibraryError=MissingOptionalLibraryError,
            AerSimulator=AerSimulator,
            have_matplotlib=have_matplotlib,
            have_pylatexenc=have_pylatexenc,
        )
    except Exception as e:
        return dict(ok=False, err=e)

def build_bb84_sample_circuit(QuantumCircuit, alice_bits, alice_bases, bob_bases, sample_size=8):
    n = min(sample_size, len(alice_bits))
    qc = QuantumCircuit(n, n)
    # Alice prepareert
    for i in range(n):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 1:  # X-basis
            qc.h(i)
    # Bob meet
    for i in range(n):
        if bob_bases[i] == 1:  # X-basis meting: rotate then Z-measure
            qc.h(i)
        qc.measure(i, i)
    return qc

# ----------------------------
# Sidebar parameters
# ----------------------------
with st.sidebar:
    st.header("Parameters")
    n_qubits = st.slider("Aantal qubits", 10, 3000, 200, step=10)
    eve_enabled = st.toggle("Voeg Eve (afluisteren) toe", value=False)
    noise = st.slider("Kanaalruis (bitflip-probabiliteit)", 0.0, 0.2, 0.02, step=0.01)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.markdown("---")
    render_mode = st.selectbox(
        "Circuit weergave",
        ["Automatisch", "Matplotlib", "Tekst"],
        help="Kies 'Tekst' als Matplotlib/pylatexenc niet beschikbaar zijn."
    )
    st.caption("Tip: zonder ruis is QBER ≈ 0% (geen Eve) of ≈ 25% (met Eve).")

# ----------------------------
# Run 1x BB84 met huidige parameters
# ----------------------------
result = run_bb84(n_qubits, eve_enabled, noise, seed)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs([
    "1) BB84-simulator",
    "2) Prestatie & QBER",
    "3) Live Qiskit-circuits",
])

# --- Tab 1: stap-voor-stap ---
with tab1:
    st.subheader("Stap-voor-stap BB84")
    colA, colB = st.columns([2, 1])
    with colA:
        df = pd.DataFrame({
            "i": np.arange(result.n_qubits),
            "Alice bit": result.alice_bits,
            "Alice basis": np.where(result.alice_bases == 0, "Z", "X"),
            "Bob basis": np.where(result.bob_bases == 0, "Z", "X"),
            "Bob raw": result.bob_raw,
            "Basis match": np.where(result.sift_mask, "✓", "✗"),
        })
        st.dataframe(df.head(200), use_container_width=True)
    with colB:
        st.metric("Sifted key length", int(result.sift_mask.sum()))
        qber_pct = (np.nan if np.isnan(result.qber) else 100 * result.qber)
        st.metric("QBER (sifted)", f"{qber_pct:.2f}%")
        if len(result.sifted_key) > 0:
            st.code("Sleutel (eerste 64 bits):\n" + ''.join(map(str, result.sifted_key[:64])))
        if eve_enabled:
            st.info("Eve is actief: verwacht ~25% QBER (zonder ruis).")
        elif noise > 0:
            st.info("Ruis actief: QBER ≈ noise in sifted key (benadering).")

# --- Tab 2: prestatie/QBER (met startknop om startup licht te houden) ---
with tab2:
    st.subheader("Prestatie & QBER-onderzoek")
    st.caption("Monte Carlo sweep over aantal qubits en ruis. Klik op 'Start sweep' om te rekenen.")
    c1, c2 = st.columns(2)
    with c1:
        sweep_qubits_max = st.slider("Sweep: maximum qubits", 50, 3000, max(500, n_qubits), step=50)
        step = st.select_slider("Stapgrootte", options=[25, 50, 100, 200], value=100)
    with c2:
        reps = st.slider("Herhalingen per punt", 1, 20, 5)
        sweep_noise = st.slider("Vaste ruis voor sweep", 0.0, 0.2, float(noise), step=0.01)
        sweep_eve = st.toggle("Eve tijdens sweep", value=eve_enabled)

    if st.button("Start sweep"):
        with st.spinner("Monte Carlo sweep bezig..."):
            rng = np.random.default_rng(seed + 123)
            qubit_grid = list(range(50, sweep_qubits_max + 1, step))
            rows = []
            for q in qubit_grid:
                qb_qber = []
                qb_len = []
                for _ in range(reps):
                    r = run_bb84(q, sweep_eve, sweep_noise, int(rng.integers(0, 1_000_000)))
                    if not np.isnan(r.qber):
                        qb_qber.append(r.qber)
                        qb_len.append(int(r.sift_mask.sum()))
                rows.append({
                    "qubits": q,
                    "mean_qber": np.mean(qb_qber) if qb_qber else np.nan,
                    "mean_keylen": np.mean(qb_len) if qb_len else 0
                })
            sweep_df = pd.DataFrame(rows)

        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(sweep_df.set_index("qubits")["mean_qber"], use_container_width=True, height=300)
            st.caption("Gemiddelde QBER vs. aantal qubits")
        with c2:
            st.line_chart(sweep_df.set_index("qubits")["mean_keylen"], use_container_width=True, height=300)
            st.caption("Gemiddelde sleutel-lengte (na sifting) vs. aantal qubits")

        st.dataframe(sweep_df, use_container_width=True)
    else:
        st.info("Klik op **Start sweep** om te starten (app blijft zo licht bij het laden).")

# --- Tab 3: live Qiskit-circuits (lazy imports + fallback) ---
with tab3:
    st.subheader("Live Qiskit-circuits (sample van eerste qubits)")
    st.caption("Deze tab laadt Qiskit pas na een klik. Zo blijft de app snel starten op Streamlit Cloud.")
    if st.button("Laad Qiskit en teken circuit"):
        with st.spinner("Qiskit laden..."):
            qisk = load_qiskit()
        if not qisk["ok"]:
            st.error(f"Kon Qiskit niet laden: {qisk['err']}")
        else:
            qc = build_bb84_sample_circuit(
                qisk["QuantumCircuit"], result.alice_bits, result.alice_bases, result.bob_bases, sample_size=8
            )

            # Beslissen hoe we renderen
            want_mpl = (
                render_mode == "Matplotlib" or
                (render_mode == "Automatisch" and qisk["have_matplotlib"] and qisk["have_pylatexenc"] and qisk["circuit_drawer"] is not None)
            )

            if want_mpl:
                try:
                    fig = qisk["circuit_drawer"](qc, output="mpl", fold=-1)
                    st.pyplot(fig)
                except qisk["MissingOptionalLibraryError"] as e:
                    st.warning(
                        "MatplotlibDrawer mist optionele dependencies. Toon tekstdiagram i.p.v. figuur.\n\n"
                        f"Detail: {e}"
                    )
                    st.text(qc.draw(output="text"))
                except Exception as e:
                    st.warning(f"Kon matplotlib niet gebruiken ({type(e).__name__}: {e}). Toon tekstdiagram.")
                    st.text(qc.draw(output="text"))
            else:
                if render_mode == "Matplotlib" and (not qisk["have_matplotlib"] or not qisk["have_pylatexenc"]):
                    st.info("Matplotlib/pylatexenc niet gevonden. Toon tekstdiagram i.p.v. figuur.")
                elif qisk["circuit_drawer"] is None:
                    st.info("Qiskit circuit_drawer niet beschikbaar. Toon tekstdiagram.")
                st.text(qc.draw(output="text"))

            # Optionele simulator-run
            if qisk["AerSimulator"] is not None and st.button("Run op AerSimulator (1024 shots)"):
                try:
                    sim = qisk["AerSimulator"]()
                    tqc = qisk["transpile"](qc, sim)
                    result_sim = sim.run(tqc, shots=1024).result()
                    counts = result_sim.get_counts()
                    st.write("Resultaat (counts):", counts)
                    st.caption("Door willekeurige bases lijken de bitstrings vaak uniform verdeeld.")
                except Exception as e:
                    st.error(f"AerSimulator niet beschikbaar of faalde: {e}")
    else:
        st.info("Klik op **Laad Qiskit en teken circuit** om de circuitweergave te genereren.")

st.markdown("---")
st.markdown(
    """
**Uitleg in het kort**  
- **BB84**: Alice encodeert bits in Z/X-bases; Bob meet in willekeurige bases. Na vergelijking behouden ze alleen posities waar de basis overeenkomt (sifted key).  
- **QBER**: Foutpercentage in de sifted key. Zonder ruis of Eve ≈ 0%. Met Eve (intercept-resend) ≈ 25%.  
- **Ruis**: Gemodelleerd als bitflip-probabiliteit op het kanaal.  
- **Circuits**: De circuitweergave laat H-poorten voor X-basiskeuze en de metingen zien.  
"""
)
