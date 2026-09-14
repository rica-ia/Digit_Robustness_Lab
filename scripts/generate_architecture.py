from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "models" / "mlp.keras"
OUTPUT_PATH = ROOT / "outputs" / "figures" / "mlp_architecture.png"

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo MLP ausente. Execute o pipeline FULL antes de gerar a figura.")
    model = tf.keras.models.load_model(MODEL_PATH)
    units = [784] + [int(layer.units) for layer in model.layers if hasattr(layer, "units")]
    names = ["Entrada", "Dense", "Dense", "Softmax"]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    xs = range(len(units))
    ax.plot(list(xs), [0] * len(units), linewidth=2)
    for x, name, unit in zip(xs, names, units):
        ax.scatter(x, 0, s=1800)
        ax.text(x, 0, f"{name}
{unit}", ha="center", va="center", fontsize=9)
    ax.set_xlim(-0.5, len(units) - 0.5); ax.set_ylim(-0.8, 0.8); ax.axis("off")
    ax.set_title("Arquitetura da MLP campeã")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"Arquitetura salva em {OUTPUT_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
