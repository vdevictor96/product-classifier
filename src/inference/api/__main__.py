""" Entrypoint of the application. """
import uvicorn


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run("src.inference.api.app:app")


if __name__ == "__main__":
    main()
