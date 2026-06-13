import argparse
import copy
import json
from pathlib import Path


DEFAULT_INPUT = "model_config.json"
DEFAULT_OUTPUT = "model_config.converted.json"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [copy.deepcopy(item) for item in value]
    if isinstance(value, dict):
        if not value:
            return []
        return [copy.deepcopy(value)]
    raise ValueError(f"Expected object or list, got {type(value).__name__}")


def _merge_candidates(default_candidates, model_config):
    if isinstance(model_config, list):
        return [copy.deepcopy(item) for item in model_config]
    if not isinstance(model_config, dict):
        raise ValueError(f"Expected model config object or list, got {type(model_config).__name__}")
    if not model_config:
        return []
    if not default_candidates:
        return [copy.deepcopy(model_config)]

    merged_candidates = []
    for default_candidate in default_candidates:
        merged = copy.deepcopy(default_candidate)
        merged.update(copy.deepcopy(model_config))
        merged_candidates.append(merged)
    return merged_candidates


def convert_chat_config(config):
    converted = copy.deepcopy(config)
    chat_config = converted.get("chat")
    if not isinstance(chat_config, dict):
        raise ValueError("model config must contain a chat object")

    stats = {
        "providers": 0,
        "settings_converted": 0,
        "models_converted": 0,
    }

    for provider, provider_config in chat_config.items():
        if not isinstance(provider_config, dict):
            raise ValueError(f"Provider {provider} config must be an object")

        stats["providers"] += 1
        default_candidates = _as_list(provider_config.get("settings", {}))
        if not isinstance(provider_config.get("settings", []), list):
            stats["settings_converted"] += 1
        provider_config["settings"] = default_candidates

        for model_name, model_config in list(provider_config.items()):
            if model_name == "settings":
                continue
            if isinstance(model_config, list):
                continue
            provider_config[model_name] = _merge_candidates(default_candidates, model_config)
            stats["models_converted"] += 1

    return converted, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert rag-llm model_config.json chat configs to candidate-list format."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input config path, default: {DEFAULT_INPUT}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output config path, default: {DEFAULT_OUTPUT}")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input config not found: {input_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output_path}. Use --force to overwrite.")

    with input_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    converted, stats = convert_chat_config(config)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Converted config written to: {output_path}")
    print(f"Providers processed: {stats['providers']}")
    print(f"Settings converted: {stats['settings_converted']}")
    print(f"Model entries converted: {stats['models_converted']}")


if __name__ == "__main__":
    main()
