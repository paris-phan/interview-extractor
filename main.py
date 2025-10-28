#!/usr/bin/env python3
"""
Main script for running the HooYouKnow interview extraction pipeline.

This script provides a simple way to run the pipeline directly without
installing the package.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src directory to Python path so we can import hooyouknow
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from hooyouknow import InterviewPipeline




def main():
    """
    Run  pipeline.
    """
    print()
    print("=" * 80)
    print("running pipeline")
    print("=" * 80)
    print()

    try:
        # Check if config files exist
        config_file = Path("config/pipeline_config.yaml")
        env_file = Path("config/.env")

        if not config_file.exists():
            print("❌ Error: config/pipeline_config.yaml not found")
            return

        if not env_file.exists():
            print("⚠️  Warning: config/.env not found")
            print("   Please copy config/.env.example to config/.env and add your API keys")
            return

        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = InterviewPipeline()
        print("✓ Pipeline initialized successfully")
        print()

        # Check for example audio files
        examples_dir = Path("examples")
        audio_files = list(examples_dir.glob("*.m4a"))

        if not audio_files:
            print("No audio files found in examples/ directory")
            print()
            print("To test the pipeline, add audio files to the examples/ directory")
            print("and create a guest_info.json file with guest information.")
            return

        print(f"Found {len(audio_files)} audio file(s) in examples/")
        print()

        # Estimate costs
        # print("Estimating costs...")
        # for audio_file in audio_files:
        #     costs = pipeline.estimate_cost(audio_path=str(audio_file))
        #     print(f"  {audio_file.name}:")
        #     print(f"    - Estimated cost: ${costs['total']:.4f}")
        # print()

        # Ask user if they want to proceed
        response = input("Would you like to process one interview? (y/n): ").strip().lower()

        if response != 'y':
            print("cancelled.")
            return

        # Process first audio file
        print()
        print("=" * 80)
        print(f"Processing: {audio_files[0].name}")
        print("=" * 80)
        print()

        # Try to get guest info
        guest_info_file = examples_dir / "guest_info.json"
        guest_name = audio_files[0].stem  # Use filename as default
        guest_title = ""

        if guest_info_file.exists():
            import json
            with open(guest_info_file) as f:
                guest_info = json.load(f)
                file_info = guest_info.get(audio_files[0].name, {})
                guest_name = file_info.get("guest_name", guest_name)
                guest_title = file_info.get("guest_title", "")

        result = pipeline.process_interview(
            audio_path=str(audio_files[0]),
            guest_name=guest_name,
            guest_title=guest_title,
        )

        # Display results
        print()
        if result["success"]:
            print("✓ Processing completed successfully!")
            print()
            print("Output files:")
            for format_name, file_path in result["output_files"].items():
                print(f"  - {format_name}: {file_path}")
            print()

            stats = result["stats"]
            print("Statistics:")
            print(f"  - Total time: {stats['timing']['total']:.2f}s")
            print(f"  - Total cost: ${stats['costs']['total']:.4f}")
            print(f"  - Total tokens: {stats['tokens']['total']:,}")
            print()
        else:
            print(f"❌ Processing failed: {result.get('error')}")

    except KeyboardInterrupt:
        print("\n\n interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print()
        print("Make sure you have:")
        print("1. Valid API keys in config/.env")
        print("2. Installed all dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
