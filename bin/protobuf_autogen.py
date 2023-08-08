from pathlib import Path
import os


def universe_dir():
    return Path(os.path.abspath(__file__)).parent.parent


def proto_dir():
    return universe_dir() / "proto"


def autogen_dir():
    return universe_dir() / "build" / "autogen"


def get_proto_files():
    files = proto_dir().glob("**/*.proto")
    return " ".join(map(str, files))


def autogen_protobuf_code():
    output_dir = str(autogen_dir())
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    command = f"protoc -I{str(proto_dir())} --cpp_out={output_dir} {get_proto_files()}"

    exit(os.system(command))


if __name__ == "__main__":
    autogen_protobuf_code()
