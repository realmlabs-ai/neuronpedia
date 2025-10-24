import logging
import time
from collections import OrderedDict
from typing import Any

from neuronpedia_inference.config import (
    Config,
    get_sae_lens_ids_from_neuronpedia_id,
    get_saelens_neuronpedia_directory_df,
)
from neuronpedia_inference.saes.saelens import SaeLensSAE  # type: ignore

logger = logging.getLogger(__name__)

# TODO: this should be in SAELens
# if we find this in the neuronpedia ID, we enable DFA
DFA_ENABLED_NP_ID_SEGMENT = "-att-"
DFA_ENABLED_NP_ID_SEGMENT_ALT = "-att_"


class SAE_TYPE:
    NEURONS = "neurons"
    SAELENS = "saelens-1"


class SAEManager:
    NEURONS_SOURCESET = "neurons"

    _instance = None  # Class variable to store the singleton instance

    @classmethod
    def get_instance(cls):
        """Get the global SAEManager instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = SAEManager()
        return cls._instance

    def __init__(
        self,
        num_layers: int = 0,
        device: str = "cuda",
    ):
        self.config = Config.get_instance()
        self.num_layers = num_layers
        self.device = device
        self.max_loaded_saes = self.config.max_loaded_saes

        self.sae_data = {}  # New consolidated dictionary
        self.sae_set_to_saes = {}
        self.valid_sae_sets = []
        self.loaded_saes = OrderedDict()  # Keep track of loaded SAEs
        # self.load_saes()

    def load_saes(self):
        server_cfg = Config.get_instance().sae_config

        self.setup_neuron_layers()

        all_sae_ids = []
        for sae_set in server_cfg:
            logger.info(f"Processing SAE set: {sae_set['set']}")
            self.valid_sae_sets.append(sae_set["set"])
            all_sae_ids.extend(sae_set["saes"])
            self.sae_set_to_saes[sae_set["set"]] = sae_set["saes"]

        starting_saes = self.get_starting_saes(all_sae_ids)

        # Load and immediately unload all SAEs not in starting_saes
        for sae_id in all_sae_ids:
            if sae_id not in starting_saes:
                self.load_sae(
                    (
                        self.config.custom_hf_model_id
                        if self.config.custom_hf_model_id
                        else self.config.model_id
                    ),
                    sae_id,
                )
                self.unload_sae(sae_id)

        # Load starting SAEs
        for sae_id in starting_saes:
            self.load_sae(
                (
                    self.config.custom_hf_model_id
                    if self.config.custom_hf_model_id
                    else self.config.model_id
                ),
                sae_id,
            )

        logger.info(f"Loaded {len(self.loaded_saes)} SAEs")

        self.print_sae_status()

    def get_starting_saes(self, all_sae_ids: list[str]) -> list[str]:
        return all_sae_ids[: (self.max_loaded_saes)]

    def load_sae(self, model_id: str, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Loading SAE: {sae_id}")

        # Check if this is a custom HF SAE by looking through the config
        is_custom_hf_sae = False
        hf_repo_id = None
        hf_sae_id = None

        for sae_set in self.config.sae_config:
            if sae_set.get("type") == "custom-hf" and sae_id in sae_set.get("saes", []):
                is_custom_hf_sae = True
                hf_repo_id = sae_set.get("hf_repo_id")
                # For custom HF SAEs, use the sae_id from config (path within repo), or default to empty string
                hf_sae_id = sae_set.get("sae_id", "")
                break

        if is_custom_hf_sae and hf_repo_id:
            # Load directly from HuggingFace
            logger.info(f"Loading custom HF SAE from {hf_repo_id}/{hf_sae_id}")
            loaded_sae, hook_name = SaeLensSAE.load(
                release=hf_repo_id,
                sae_id=hf_sae_id,
                device=self.device,
                dtype=self.config.sae_dtype,
            )
        else:
            # Load from SAELens directory
            sae_lens_release, sae_lens_id = get_sae_lens_ids_from_neuronpedia_id(
                model_id=model_id,
                neuronpedia_id=sae_id,
                df_exploded=get_saelens_neuronpedia_directory_df(),
            )

            loaded_sae, hook_name = SaeLensSAE.load(
                release=sae_lens_release,
                sae_id=sae_lens_id,
                device=self.device,
                dtype=self.config.sae_dtype,
            )

        self.sae_data[sae_id] = {
            "sae": loaded_sae,
            "hook": hook_name,
            "neuronpedia_id": loaded_sae.cfg.metadata.neuronpedia_id,
            "type": SAE_TYPE.SAELENS,
            # TODO: this should be in SAELens
            "dfa_enabled": (
                loaded_sae.cfg.metadata.neuronpedia_id is not None
                and (
                    DFA_ENABLED_NP_ID_SEGMENT in loaded_sae.cfg.metadata.neuronpedia_id
                    or DFA_ENABLED_NP_ID_SEGMENT_ALT
                    in loaded_sae.cfg.metadata.neuronpedia_id
                )
            ),
            "transcoder": False,  # You might want to set this based on some condition
        }

        self.loaded_saes[sae_id] = None  # We're using OrderedDict as an OrderedSet
        if len(self.loaded_saes) > self.max_loaded_saes:
            lru_sae = next(iter(self.loaded_saes))
            self.unload_sae(lru_sae)

        end_time = time.time()

        logger.info(
            f"Successfully loaded SAE: {sae_id} in {end_time - start_time:.2f} seconds"
        )

    def unload_sae(self, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Starting to unload SAE: {sae_id}")

        if sae_id in self.sae_data:
            self.sae_data[sae_id]["sae"] = None

        if sae_id in self.loaded_saes:
            del self.loaded_saes[sae_id]

        end_time = time.time()
        logger.info(
            f"Successfully unloaded SAE: {sae_id} in {end_time - start_time:.2f} seconds"
        )

    def get_sae(self, source: str) -> Any:
        if source not in self.loaded_saes:
            self.load_sae(
                (
                    self.config.custom_hf_model_id
                    if self.config.custom_hf_model_id
                    else self.config.model_id
                ),
                source,
            )
        else:
            self.loaded_saes.move_to_end(source)
        return self.sae_data.get(source, {}).get("sae")

    def setup_neuron_layers(self):
        neurons_sourceset = []
        for layer in range(self.num_layers):
            layer_str = str(layer)
            neurons_sourceset.append(layer_str)
            self.sae_data[layer_str] = {
                "sae": None,
                "neuronpedia_id": None,
                "dfa_enabled": False,
                "transcoder": False,
                "type": SAE_TYPE.NEURONS,
                "hook": f"blocks.{layer_str}.mlp.hook_post",
            }

        self.sae_set_to_saes[self.NEURONS_SOURCESET] = neurons_sourceset
        return neurons_sourceset

    def print_sae_status(self):
        """
        Print a nicely formatted status of loadable and loaded SAEs.
        """
        print("\nSAE Status:")
        print("===========")

        print("\nLoadable SAEs:")
        for sae_set, sae_ids in self.sae_set_to_saes.items():
            if sae_set == self.NEURONS_SOURCESET:
                continue
            print(f"  {sae_set}:")
            for sae_id in sae_ids:
                status = "Loaded" if sae_id in self.loaded_saes else "Not Loaded"
                print(f"    - {sae_id}: {status}")

        print("\nCurrently Loaded SAEs:")
        for i, sae_id in enumerate(self.loaded_saes, 1):
            print(f"  {i}. {sae_id}")

        print(f"\nTotal Loaded: {len(self.loaded_saes)} / {self.max_loaded_saes}")

    # Utility methods
    def get_sae_type(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("type")

    def get_sae_hook(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("hook")

    def is_dfa_enabled(self, sae_id: str) -> bool:
        return self.sae_data.get(sae_id, {}).get("dfa_enabled", False)

    def get_valid_sae_sets(self):
        return self.valid_sae_sets
