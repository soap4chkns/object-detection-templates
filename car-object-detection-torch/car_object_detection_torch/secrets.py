from abc import ABC, abstractmethod


class SecretRetrieval(ABC):
    @abstractmethod
    def get_secret(self) -> str:
        # retrieve api key by environment (i.e. google colab, local)
        pass


class LocalRetrieval(SecretRetrieval):
    def get_secret(self) -> str:
        # credentials don't exist in local environment
        return "123"


class GoogleColabRetrieval(SecretRetrieval):
    def get_secret(self) -> str:
        try:
            # only available in the colab environment
            from google.colab import userdata  # type: ignore

            api_key = userdata.get("WANDB_API_KEY")
            return api_key
        except ImportError:
            raise ImportError(
                "google.colab.userdata is only available in google colab environment"
            )
        except Exception:
            raise ValueError("Secret not found")


secrets_factory: dict[str, SecretRetrieval] = {
    "local": LocalRetrieval(),
    "google_colab": GoogleColabRetrieval(),
}
