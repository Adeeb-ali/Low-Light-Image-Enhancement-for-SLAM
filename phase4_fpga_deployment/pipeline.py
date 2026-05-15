from quantizer import run_quantization
from compile import compile_model
from report_generator import (
    analyze_onnx,
    analyze_xmodel,
    analyze_arch,
    analyze_environment,
    save_report
)


def main():

    print("="*60)
    print("STEP 1: QUANTIZATION (PTQ)")
    print("="*60)
    run_quantization()

    print("="*60)
    print("STEP 2: COMPILATION")
    print("="*60)
    compile_model()

    print("="*60)
    print("STEP 3: REPORT")
    print("="*60)

    onnx_r = analyze_onnx()
    xmodel_r = analyze_xmodel()
    arch_r = analyze_arch()
    env_r = analyze_environment()

    save_report(onnx_r, xmodel_r, arch_r, env_r)


if __name__ == "__main__":
    main()