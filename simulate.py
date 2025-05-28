#!/usr/bin/env python3
import yaml

def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    print("Running hybrid_prime_predicter simulation with config:", cfg)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
