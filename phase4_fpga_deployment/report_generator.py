import os
import json
import onnx
import xir
import datetime
import subprocess


# =========================
# ONNX ANALYSIS (optional)
# =========================
def analyze_onnx():
    if not os.path.exists("output"):
        return {"warning": "output folder not found"}

    for f in os.listdir("output"):
        if f.endswith(".onnx"):
            model_path = os.path.join("output", f)
            model = onnx.load(model_path)

            op_counts = {}
            for node in model.graph.node:
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

            return {
                "file": f,
                "total_nodes": len(model.graph.node),
                "operations": op_counts,
                "opset": model.opset_import[0].version if model.opset_import else "unknown"
            }

    return {"warning": "No ONNX file found"}


# =========================
# XMODEL ANALYSIS
# =========================
def analyze_xmodel():

    if not os.path.exists("compiled_model"):
        return {"error": "compiled_model folder not found"}

    report = {
        "subgraphs": [],
        "fingerprint_match": False
    }

    for root, _, files in os.walk("compiled_model"):
        for f in files:
            if f.endswith(".xmodel"):
                xmodel_path = os.path.join(root, f)
                report["compiled_model_path"] = xmodel_path

                try:
                    graph = xir.Graph.deserialize(xmodel_path)
                    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

                    for sub in subgraphs:
                        device = "CPU"
                        if sub.has_attr("device") and sub.get_attr("device") == "DPU":
                            device = "DPU"

                        sub_info = {
                            "name": sub.get_name(),
                            "device": device,
                            "ops": []
                        }

                        try:
                            sub_info["ops"] = [op.get_type() for op in sub.get_ops()]
                        except:
                            pass

                        report["subgraphs"].append(sub_info)

                except Exception as e:
                    report["error"] = str(e)

                return report

    return {"warning": "No compiled .xmodel found"}


# =========================
# ARCH CHECK
# =========================
def analyze_arch():

    arch_path = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"

    if not os.path.exists(arch_path):
        return {"error": "arch.json not found (are you inside Docker?)"}

    try:
        with open(arch_path) as f:
            arch_data = json.load(f)

        return {
            "arch_file": arch_path,
            "target": arch_data.get("target", "unknown"),
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# ENVIRONMENT CHECK
# =========================
def analyze_environment():

    env = {
        "timestamp": str(datetime.datetime.now()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    }

    try:
        result = subprocess.run(
            ["pip", "show", "pytorch-nndct"],
            capture_output=True,
            text=True
        )

        for line in result.stdout.splitlines():
            if line.startswith("Version"):
                env["pytorch_nndct"] = line.split(":")[-1].strip()

    except:
        env["pytorch_nndct"] = "unknown"

    return env


# =========================
# FINAL REPORT
# =========================
def save_report(onnx_report, xmodel_report, arch_report, env_report):

    os.makedirs("report", exist_ok=True)

    full = {
        "environment": env_report,
        "arch": arch_report,
        "onnx": onnx_report,
        "xmodel": xmodel_report,
    }

    # Deployment decision
    has_dpu = any(
        sub.get("device") == "DPU"
        for sub in xmodel_report.get("subgraphs", [])
    )

    full["DEPLOYMENT_VERDICT"] = {
        "dpu_detected": has_dpu,
        "verdict": "✅ READY (DPU detected)" if has_dpu
                   else "⚠️ CPU fallback detected"
    }

    path = "report/final_report.json"

    with open(path, "w") as f:
        json.dump(full, f, indent=4)

    print(f"\n📄 Report saved → {path}")
    print("\n", full["DEPLOYMENT_VERDICT"]["verdict"])