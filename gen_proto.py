"""Генерация Python-клиента gRPC из proto-контрактов T-Invest API (без официального SDK)."""
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).parent
PROTO_DIR = ROOT / "contracts"
OUT_DIR = ROOT / "tinvest_pb"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    # include на grpc_tools/_proto нужен, чтобы protoc нашёл google/protobuf/*.proto (timestamp, descriptor)
    import grpc_tools
    std_include = str(Path(grpc_tools.__file__).parent / "_proto")
    files = ["marketdata.proto", "common.proto", "google/api/field_behavior.proto"]
    args = [
        "protoc",
        f"-I{PROTO_DIR}",
        f"-I{std_include}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *[str(PROTO_DIR / f) for f in files],
    ]
    code = protoc.main(args)
    if code != 0:
        raise SystemExit(f"protoc завершился с кодом {code}")
    print("Сгенерировано в", OUT_DIR)


if __name__ == "__main__":
    main()
