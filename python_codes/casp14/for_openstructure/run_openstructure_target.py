from pathlib import Path
import docker


# Set the necessary constants. The available predictor keys are the following:
# AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
PREDICTOR_KEY = "TS427"
TARGET_DIR = Path(f"../workdir/casp14/for_openstructure/{PREDICTOR_KEY}_results").resolve()


def main():

    docker_client = docker.from_env()

    current_dir = Path(".").resolve()
    print(current_dir)

    docker_client.containers.run(
        "4d90c60d1ad2",
        "openstructure_target.py",
        auto_remove=True,
        volumes=[
            f"{current_dir}:/home",
            f"{TARGET_DIR}:/data"
        ]
    )


if __name__ == "__main__":
    main()
