# TODO: this is supposed to generate dashboards for an SAELens SAE, but it's not currently repurposed yet. It's currently identical to the vector script.
# TODO: include saelensid, saelensconfig, saelensRelease
#
# This generates dashboards for your SAE or set of vectors for a model, for direct import into Neuronpedia.
# Once the outputs are generated, they can be uploaded to the neuronpedia-datasets S3 bucket, where anyone can import and experiment with them.
#
# It uses typer, so you can do --help to see the available arguments.
# If a required argument is not provided, it will prompt you for it.
# Usage: python generate-dashboards.py [arguments]
#
# Example: Axbench
#
# python generate-dashboards.py \
#     --creator-name='AxBench Team' \
#     --release-id=axbench \
#     --release-title='AxBench Paper' \
#     --url=https://github.com/stanfordnlp/axbench \
#     --model-name=gemma-2-2b-it \
#     --model-dtype=bfloat16 \
#     --neuronpedia-source-set-id=axbench-reft-r1-res-16k \
#     --neuronpedia-source-set-description='Residual Stream - 16k' \
#     --hf-weights-repo-id=pyvene/gemma-reft-r1-2b-it-res \
#     --hf-weights-path=l20/weight.pt \
#     --hook-point=hook_resid_pre \
#     --source-dtype=bfloat16 \
#     --layer-num=20 \
#     --prompts-huggingface-dataset-path=monology/pile-uncopyrighted \
#     --n-prompts-total=8192 \
#     --n-tokens-in-prompt=128 \
#     --n-prompts-per-batch=128 \
#     --include-original-vectors-in-output \
#     --prepend-chat-template-text='<bos><start_of_turn>user\nWrite me a random sentence.<end_of_turn>\n<start_of_turn>model\n' \
#     --activation-thresholds-json-file=activation-thresholds.example.json


import json
from typing import List, Optional, Any
from datetime import datetime
from cuid2 import Cuid
import os
import gzip
import typer
import shutil
from typing import Annotated
from neuronpedia_utils.db_models.source_release import SourceRelease
from neuronpedia_utils.db_models.model import Model
from neuronpedia_utils.db_models.source_set import SourceSet
from neuronpedia_utils.db_models.source import Source
from neuronpedia_utils.db_models.activation import Activation
from neuronpedia_utils.db_models.feature import Feature
import dotenv
from enum import Enum
from huggingface_hub import hf_hub_download
import torch
from safetensors import safe_open
from sae_dashboard.neuronpedia.vector_set import VectorSet
from sae_dashboard.neuronpedia.neuronpedia_vector_runner import (
    NeuronpediaVectorRunner,
    NeuronpediaVectorRunnerConfig,
)

dotenv.load_dotenv(".env.default")
dotenv.load_dotenv()

OUTPUT_DIR = "./exports"
# directory used for intermediate outputs (from saedashboard, before conversion to neuronpedia format)
INTERMEDIATE_OUTPUT_DIR = "./exports-processing"

creator_id = os.getenv("DEFAULT_CREATOR_ID")
if creator_id is None or creator_id == "":
    raise ValueError("DEFAULT_CREATOR_ID is not set")

DEFAULT_CREATOR_ID = creator_id

CUID_GENERATOR: Cuid = Cuid(length=25)

created_at = datetime.now()

CACHED_ACTIVATIONS_DIR = "./cached_activations"


class HOOK_POINT_TYPE_CHOICES(str, Enum):
    hook_resid_pre = "hook_resid_pre"
    hook_resid_mid = "hook_resid_mid"
    hook_resid_post = "hook_resid_post"
    hook_mlp_out = "hook_mlp_out"
    hook_attn_in = "hook_attn_in"
    hook_attn_out = "hook_attn_out"
    hook_z = "hook_z"


app = typer.Typer()


def make_option(*option_names: str, help_text: str, **kwargs) -> Any:
    """Create a Typer Option with the same text for both help and prompt."""
    return typer.Option(
        *option_names,
        help=help_text,
        prompt="\n" + help_text + "\n",
        **kwargs,
    )


@app.command()
def main(
    ctx: typer.Context,
    creator_name: Annotated[
        str,
        make_option(
            "--creator-name",
            help_text="[Author] Name: Name of the creator (e.g., your organization/team name).",
        ),
    ],
    release_id: Annotated[
        str,
        make_option(
            "--release-id",
            help_text="[Release] Release ID: Enter the release id (eg gemma-scope). Must be alphanumeric, but can also include dashes. Your release will be available at https://[neuronpedia_domain]/[release_id]",
        ),
    ],
    release_title: Annotated[
        str,
        make_option(
            "--release-title",
            help_text="[Release] Release Title: Human-readable description of the release - probably a shortened version of your paper. Eg Exploring Gemma 2 with Gemma Scope.",
        ),
    ],
    url: Annotated[
        str,
        make_option(
            "--url",
            help_text="[Info] URL: URL associated with your paper/release. Include https://...",
        ),
    ],
    model_name: Annotated[
        str,
        make_option(
            "--model-name",
            help_text="[Model] Model Name: The TransformerLens name of the model to be used for the dashboard. Eg 'gemma-2-2b-it'. See https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html",
        ),
    ],
    model_dtype: Annotated[
        str,
        make_option(
            "--model-dtype",
            help_text="[Model] Model DType: The dtype of the model to be used for dashboard generation. Eg 'float32', 'float16', 'bfloat16'.",
        ),
    ],
    neuronpedia_source_set_id: Annotated[
        str,
        make_option(
            "--neuronpedia-source-set-id",
            help_text="[Source] Neuronpedia Source Set ID: All dashboards on Neuronpedia belong to a 'Source Set', which is an identifier. Please specify what the 'source set id' is for your data - it should be short and descriptive of the author, hook, and optionally, width (number of features/vectors per layer).\nFor example, an example source set ID is gemmascope-res-16k. The URL is then https://[neuronpedia_domain]/gemma-2-2b/gemmascope-res-16k.\nDo not include the layer number - that will automatically be prepended later.",
        ),
    ],
    neuronpedia_source_set_description: Annotated[
        str,
        make_option(
            "--neuronpedia-source-set-description",
            help_text="[Source] Neuronpedia Source Set Description: When this source set is displayed on Neuronpedia, this is the description that will be shown. Usually, it is a short human-readable hook and width for this source. Eg Residual Stream - 16k",
        ),
    ],
    hf_weights_repo_id: Annotated[
        str,
        make_option(
            "--hf-weights-repo-id",
            help_text="[Source] HuggingFace Repository ID: Huggingface repository ID for your weights/data in the form [user]/[repo_id], NOT INCLUDING the folder. Eg 'google/gemma-scope-2b-pt-res'",
        ),
    ],
    hf_weights_path: Annotated[
        str,
        make_option(
            "--hf-weights-path",
            help_text="[Source] HuggingFace Weights Path: Path to the weights on HuggingFace in the form 'layer_0/width_16k/average_l0_105/weights.pt'. Do not include the repo name.",
        ),
    ],
    hook_point: Annotated[
        HOOK_POINT_TYPE_CHOICES,
        make_option(
            "--hook-point",
            help_text=f"[Source] Hook Point: The TransformerLens hook point to use for the dashboard. Must be one of: {', '.join([f'{choice}' for choice in HOOK_POINT_TYPE_CHOICES])}.",
        ),
    ],
    source_dtype: Annotated[
        str,
        make_option(
            "--source-dtype",
            help_text="[Source] Source DType: The dtype of the vectors/SAEs/features weights. Eg 'float32', 'float16', 'bfloat16'.",
        ),
    ],
    layer_num: Annotated[
        int,
        make_option(
            "--layer-num",
            help_text="[Source] Layer Number: The layer number that this source/SAE is trained on. Eg 20.",
        ),
    ],
    prompts_huggingface_dataset_path: Annotated[
        str,
        make_option(
            "--prompts-huggingface-dataset-path",
            help_text="[Dashboard Gen Parameters] HuggingFace Dataset Path: The path to the HuggingFace dataset to use for prompts. Eg 'monology/pile-uncopyrighted'.",
        ),
    ],
    n_prompts_total: Annotated[
        int,
        make_option(
            "--n-prompts-total",
            help_text="[Dashboard Gen Parameters] Total Prompts: The number of prompts to use to generate activations for the dashboard. More will give you a wider breadth of activations, but requires more time and memory. 16,384 or 24,576 are common values.",
        ),
    ] = 24576,
    n_tokens_in_prompt: Annotated[
        int,
        make_option(
            "--n-tokens-in-prompt",
            help_text="[Dashboard Gen Parameters] Context Tokens per Prompt: The number of tokens per prompt to use for each activation in the dashboard. More requires more time and memory. We typically use 128.",
        ),
    ] = 128,
    n_prompts_per_batch: Annotated[
        int,
        make_option(
            "--n-prompts-per-batch",
            help_text="[Dashboard Gen Parameters] Prompts per Batch (performance): When generating activations, how many prompts should we show the GPU in parallel? This is constrained by GPU memory, using the default of 128 is a good starting point.",
        ),
    ] = 128,
    include_original_vectors_in_output: Annotated[
        bool,
        make_option(
            "--include-original-vectors-in-output/--no-include-original-vectors-in-output",
            help_text="[Dashboard Gen Parameters] Include Original Vectors in Output: Whether to include the original vectors of each feature in the output. This will make your outputs larger, but will allow you to see the original vectors in the dashboard.",
        ),
    ] = True,
    prepend_chat_template_text: Annotated[
        Optional[str],
        typer.Option(
            "--prepend-chat-template-text",
            help="[Optional] [Dashboard Gen Parameters] Prepend Chat Template Text: If you are generating activations for an instruct model using a dataset that does not have chat templates, you can prepend each prompt with a chat template like:'<bos><start_of_turn>user\\nWrite me a random sentence.<end_of_turn>\\n<start_of_turn>model\\n'",
        ),
    ] = None,
):
    # Set activation_thresholds_json_file to None since we removed the parameter
    activation_thresholds_json_file = None
    
    print("Running with arguments:\n")
    for param, value in ctx.params.items():
        print(f"{param}: {value}")

    print("--------------------------------")
    print("Equivalent command is:")

    command = "python generate-dashboards.py"
    for name, value in ctx.params.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    command += f" --{name.replace('_', '-')}"
            else:
                # Quote strings if they contain spaces
                if isinstance(value, str) and (" " in value or "'" in value):
                    value = f"'{value}'"
                command += f" --{name.replace('_', '-')}={value}"
    command = command.replace(" --", " \\\n    --")
    print(command)

    try:
        if os.path.exists(CACHED_ACTIVATIONS_DIR):
            shutil.rmtree(CACHED_ACTIVATIONS_DIR)

        print("Downloading weights from HuggingFace...")
        path_to_weights = hf_hub_download(
            repo_id=hf_weights_repo_id, filename=hf_weights_path
        )
        if path_to_weights is None:
            raise ValueError(
                f"Failed to download weights at {hf_weights_repo_id}/{hf_weights_path} from HuggingFace. Please check that it is valid and that if needed, you've added your HF_TOKEN to the environment variables."
            )

        if activation_thresholds_json_file is not None:
            activation_thresholds = read_json_file(activation_thresholds_json_file)
            # ensure the thresholds are type: dict[str, float | int]
            if not isinstance(activation_thresholds, dict):
                raise ValueError(
                    f"Activation thresholds file {activation_thresholds_json_file} must be a dictionary mapping from index (string) to threshold (float or int)."
                )
            for key, value in activation_thresholds.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Activation thresholds file {activation_thresholds_json_file} must be a dictionary mapping from index (string) to threshold (float or int)."
                    )
                if not isinstance(value, (float, int)):
                    raise ValueError(
                        f"Activation thresholds file {activation_thresholds_json_file} must be a dictionary mapping from index (string) to threshold (float or int)."
                    )
            # convert the string keys to ints
            activation_thresholds = {
                int(key): value for key, value in activation_thresholds.items()
            }
        else:
            activation_thresholds = None

        # Load safetensors file
        if path_to_weights.endswith('.safetensors'):
            with safe_open(path_to_weights, framework="pt", device="cpu") as f:
                # Get all tensor names and load them into a dict
                weights = {}
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
                # If there's only one tensor, use it directly
                if len(weights) == 1:
                    weights = list(weights.values())[0]
        else:
            weights = torch.load(path_to_weights, weights_only=False)

        hook_point_name = f"blocks.{layer_num}.{hook_point.value}"

        vector_set = VectorSet(
            vectors=weights,
            hook_point=hook_point_name,
            hook_layer=layer_num,
            model_name=model_name,
            hook_head_index=None,
            names=[],
            prepend_bos=False,
        )

        intermediate_output_dir = (
            f"{INTERMEDIATE_OUTPUT_DIR}/{model_name}/{neuronpedia_source_set_id}"
        )
        shutil.rmtree(intermediate_output_dir)
        os.makedirs(intermediate_output_dir)

        print("Generating activations...")

        runner_config = NeuronpediaVectorRunnerConfig(
            outputs_dir=intermediate_output_dir,
            huggingface_dataset_path=prompts_huggingface_dataset_path,
            vector_dtype=source_dtype,
            model_dtype=model_dtype,
            n_prompts_total=n_prompts_total,
            n_tokens_in_prompt=n_tokens_in_prompt,
            n_prompts_in_forward_pass=n_prompts_per_batch,
            n_vectors_at_a_time=n_prompts_per_batch,
            use_wandb=False,
            include_original_vectors_in_output=include_original_vectors_in_output,
            prepend_chat_template_text=(
                prepend_chat_template_text.replace("\\n", "\n")
                if prepend_chat_template_text is not None
                else None
            ),
            activation_thresholds=activation_thresholds,
        )

        runner = NeuronpediaVectorRunner(vector_set, runner_config)
        runner.run()

        print("Activations generated successfully.")

        print("Converting to Neuronpedia format for final output...")

        global VECTOR_STEER_HOOK_NAME
        VECTOR_STEER_HOOK_NAME = hook_point.value

        # get the hf folder id from the hf weights path
        hf_folder_id = "/".join(hf_weights_path.split("/")[:-1])

        final_output_dir = ""

        intermediate_output_dir_subdir = os.path.join(
            intermediate_output_dir, f"{model_name}_{hook_point_name}"
        )

        for file in sorted(
            f
            for f in os.listdir(intermediate_output_dir_subdir)
            if f.startswith("batch-")
        ):
            print("reading activations from batch file", file)
            batch_data = read_json_file(
                os.path.join(intermediate_output_dir_subdir, file)
            )
            global OUTPUT_PATH_BASE

            OUTPUT_PATH_BASE = f"{OUTPUT_DIR}/{model_name}"
            if not os.path.exists(OUTPUT_PATH_BASE):
                os.makedirs(OUTPUT_PATH_BASE)

            source_suffix = batch_data["sae_id_suffix"]

            source_id = (
                str(layer_num)
                + "-"
                + neuronpedia_source_set_id
                + ("__" + source_suffix if source_suffix else "")
            )

            final_output_dir = os.path.join(OUTPUT_PATH_BASE, source_id)
            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)

            # make the release jsonl
            release_file_path = os.path.join(final_output_dir, "release.jsonl")
            with open(release_file_path, "w") as f:
                release = SourceRelease(
                    name=release_id,
                    description=release_title,
                    descriptionShort=release_title,
                    urls=[url] if url else [],
                    creatorNameShort=creator_name,
                    creatorName=creator_name,
                    creatorId=DEFAULT_CREATOR_ID,
                    createdAt=created_at,
                )
                f.write(json.dumps(release.__dict__, default=datetime_handler) + "\n")

            # make the model jsonl
            model_file_path = os.path.join(final_output_dir, "model.jsonl")
            with open(model_file_path, "w") as f:
                model = Model(
                    id=model_name,
                    instruct=model_name.endswith("-it"),
                    displayNameShort=model_name,
                    displayName=model_name,
                    creatorId=DEFAULT_CREATOR_ID,
                    createdAt=created_at,
                    updatedAt=created_at,
                )
                f.write(json.dumps(model.__dict__, default=datetime_handler) + "\n")

            # make the sourceset jsonl
            sourceset_file_path = os.path.join(final_output_dir, "sourceset.jsonl")
            with open(sourceset_file_path, "w") as f:
                sourceset = SourceSet(
                    modelId=model_name,
                    name=neuronpedia_source_set_id,
                    creatorId=DEFAULT_CREATOR_ID,
                    createdAt=created_at,
                    creatorName=creator_name,
                    releaseName=release_id,
                    description=neuronpedia_source_set_description,
                    visibility="PUBLIC",
                )
                f.write(json.dumps(sourceset.__dict__, default=datetime_handler) + "\n")

            # make the source jsonl
            source_file_path = os.path.join(final_output_dir, "source.jsonl")
            with open(source_file_path, "w") as f:
                source = Source(
                    modelId=model_name,
                    setName=neuronpedia_source_set_id,
                    visibility="PUBLIC",
                    dataset=prompts_huggingface_dataset_path,
                    id=source_id,
                    num_prompts=n_prompts_total,
                    num_tokens_in_prompt=n_tokens_in_prompt,
                    hfRepoId=hf_weights_repo_id,
                    hfFolderId=hf_folder_id,
                    creatorId=DEFAULT_CREATOR_ID,
                )
                f.write(json.dumps(source.__dict__, default=datetime_handler) + "\n")
            process_data(
                batch_data, neuronpedia_source_set_id, file.replace(".json", "")
            )

        print(
            "\n\n ==================== Dashboards generated successfully. ==================== \n"
        )
        print("The dashboards are available in the source output directory:")
        print(final_output_dir)

    except BaseException as e:
        print(f"\nError: {e}")
        print("\nTo run this job again, use this command (fixing any errors first):\n")

        print(command)
        raise typer.Abort()


def read_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def process_data(batch_data, source_set_name, batch_file_name):
    source_set = source_set_name
    model_id = batch_data["model_id"]
    layer_num = batch_data["layer"]
    source_suffix = batch_data["sae_id_suffix"]

    source_id = (
        str(layer_num)
        + "-"
        + source_set
        + ("__" + source_suffix if source_suffix else "")
    )

    source_dir = os.path.join(OUTPUT_PATH_BASE, source_id)

    activations: List[Activation] = []
    features: List[Feature] = []

    for feature_data in batch_data["features"]:
        new_feature = Feature(
            modelId=model_id,
            layer=source_id,
            index=feature_data["feature_index"],
            creatorId=DEFAULT_CREATOR_ID,
            createdAt=created_at,
            hasVector="vector" in feature_data and len(feature_data["vector"]) > 0,
            vector=feature_data["vector"] if "vector" in feature_data else [],
            vectorLabel=None,
            hookName=(
                f"blocks.{layer_num}.{VECTOR_STEER_HOOK_NAME}"
                if "vector" in feature_data
                else None
            ),
            topkCosSimIndices=[],
            topkCosSimValues=[],
            neuron_alignment_indices=feature_data["neuron_alignment_indices"],
            neuron_alignment_values=feature_data["neuron_alignment_values"],
            neuron_alignment_l1=feature_data["neuron_alignment_l1"],
            correlated_neurons_indices=feature_data["correlated_neurons_indices"],
            correlated_neurons_pearson=feature_data["correlated_neurons_pearson"],
            correlated_neurons_l1=feature_data["correlated_neurons_l1"],
            correlated_features_indices=feature_data["correlated_features_indices"],
            correlated_features_pearson=feature_data["correlated_features_pearson"],
            correlated_features_l1=feature_data["correlated_features_l1"],
            neg_str=feature_data["neg_str"],
            neg_values=feature_data["neg_values"],
            pos_str=feature_data["pos_str"],
            pos_values=feature_data["pos_values"],
            frac_nonzero=feature_data["frac_nonzero"],
            freq_hist_data_bar_heights=feature_data["freq_hist_data_bar_heights"],
            freq_hist_data_bar_values=feature_data["freq_hist_data_bar_values"],
            logits_hist_data_bar_heights=feature_data["logits_hist_data_bar_heights"],
            logits_hist_data_bar_values=feature_data["logits_hist_data_bar_values"],
            decoder_weights_dist=feature_data["decoder_weights_dist"],
        )
        max_act_approx = 0
        for activation_data in feature_data["activations"]:
            max_value = max(activation_data["values"])
            max_value_token_index = activation_data["values"].index(max_value)
            if max_value > max_act_approx:
                max_act_approx = max_value
            new_activation = Activation(
                id=CUID_GENERATOR.generate(),
                tokens=activation_data["tokens"],
                modelId=model_id,
                layer=source_id,
                index=feature_data["feature_index"],
                maxValue=max_value,
                maxValueTokenIndex=max_value_token_index,
                minValue=min(activation_data["values"]),
                values=activation_data["values"],
                dfaValues=(
                    activation_data["dfa_values"]
                    if "dfa_values" in activation_data
                    else []
                ),
                dfaTargetIndex=(
                    activation_data["dfa_targetIndex"]
                    if "dfa_targetIndex" in activation_data
                    else None
                ),
                dfaMaxValue=(
                    activation_data["dfa_maxValue"]
                    if "dfa_maxValue" in activation_data
                    else None
                ),
                creatorId=DEFAULT_CREATOR_ID,
                createdAt=created_at,
                lossValues=(
                    activation_data["loss_values"]
                    if "loss_values" in activation_data
                    else []
                ),
                logitContributions=(
                    activation_data["logit_contributions"]
                    if "logit_contributions" in activation_data
                    else None
                ),
                binContains=activation_data["bin_contains"],
                binMax=activation_data["bin_max"],
                binMin=activation_data["bin_min"],
                qualifyingTokenIndex=activation_data["qualifying_token_index"],
                dataIndex=None,
                dataSource=None,
            )

            activations.append(new_activation)
        new_feature.maxActApprox = max_act_approx
        if "vector" in feature_data:
            new_feature.vectorDefaultSteerStrength = new_feature.maxActApprox
        features.append(new_feature)

    # make features directory
    features_dir = os.path.join(source_dir, "features")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # write the features to a jsonl
    features_file_path = os.path.join(features_dir, f"{batch_file_name}.jsonl")
    with open(features_file_path, "w") as f:
        for feature in features:
            f.write(json.dumps(feature.__dict__, default=datetime_handler) + "\n")

    # gzip the file
    with open(features_file_path, "rb") as f_in:
        with open(features_file_path + ".gz", "wb") as f_out:
            f_out.write(gzip.compress(f_in.read(), compresslevel=5))
    os.remove(features_file_path)

    activations_dir = os.path.join(source_dir, "activations")
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    activations_file_path = os.path.join(activations_dir, f"{batch_file_name}.jsonl")
    with open(activations_file_path, "w") as f:
        for activation in activations:
            f.write(json.dumps(activation.__dict__, default=datetime_handler) + "\n")

    # gzip the file
    with open(activations_file_path, "rb") as f_in:
        with open(activations_file_path + ".gz", "wb") as f_out:
            f_out.write(gzip.compress(f_in.read(), compresslevel=5))
    os.remove(activations_file_path)


if __name__ == "__main__":
    app()
