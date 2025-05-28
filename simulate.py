#!/usr/bin/env python3
import yaml

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    print("Running simulation with config:", cfg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the prime discovery simulation")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
