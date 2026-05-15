import subprocess
import os

ARCH_JSON = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"


def find_xmodel():
    if not os.path.exists("output"):
        return None

    for f in os.listdir("output"):
        if f.endswith(".xmodel"):
            return f

    return None


def compile_model():

    xmodel = find_xmodel()

    if xmodel is None:
        raise RuntimeError("❌ No .xmodel found in output/")

    xmodel_path = os.path.join("output", xmodel)

    os.makedirs("compiled_model", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    cmd = [
        "vai_c_xir",
        "-x", xmodel_path,
        "-a", ARCH_JSON,
        "-o", "compiled_model",
        "-n", "net"
    ]

    print("🔹 Running compiler...")
    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    with open("report/compile_log.txt", "w") as f:
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("❌ Compilation failed (check report/compile_log.txt)")

    for f in os.listdir("compiled_model"):
        if f.endswith(".xmodel"):
            path = os.path.join("compiled_model", f)
            print(f"✅ Compiled model → {path}")
            return path

    raise RuntimeError("❌ No compiled model found")