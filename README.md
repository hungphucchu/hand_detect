# Hand Detect

A hand gesture dect and recognition project using **efficientnet_b0**. Supports training from scratch or using pretrained models.

---

## Setup

1. **Create a virtual environment** (Python 3.11 recommended):

```bash
python3.11 -m venv tf-env
````

2. **Activate the environment**:

```bash
# macOS / Linux
source tf-env/bin/activate

# Windows (PowerShell)
.\tf-env\Scripts\Activate.ps1
```

3. **Install dependencies**:

```bash
pip install torchvision torch numpy matplotlib albumentations
```

> You may add other dependencies as needed (`torch`, `numpy`, etc.).

---

## Usage

### Train the model from scratch

```bash
python main.py --train
```

### Run inference with a saved model to test

```bash
python main.py --test
```

---

## Notes

* The model saves the best checkpoint as `best_facenet_model.pth`.

---

## License

MIT License