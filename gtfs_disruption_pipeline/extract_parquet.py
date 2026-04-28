"""
Extract parquet files from zipped feeds before ingestion.
"""
import zipfile
import os
from pathlib import Path
from typing import List, Tuple


def extract_all_parquet(
    zip_dir: str = ".",
    output_dir: str = "extracted_parquet",
    pattern: str = "*_files_list.zip"
) -> List[Tuple[str, str]]:
    """
    Extract all parquet files from zipped feeds.
    
    Parameters
    ----------
    zip_dir : str
        Directory containing zip files
    output_dir : str
        Directory to extract to
    pattern : str
        Glob pattern for zip files
    
    Returns
    -------
    List of (zip_path, extracted_dir) tuples
    """
    zip_path = Path(zip_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extracted = []
    
    for zip_file in sorted(zip_path.glob(pattern)):
        print(f"Extracting {zip_file.name}...")
        
        # Create subdirectory for this zip
        extract_dir = output_path / zip_file.stem.replace('_files_list', '')
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                # Extract only parquet files
                for member in z.namelist():
                    if member.endswith('.parquet'):
                        z.extract(member, extract_dir)
                        print(f"  - {member}")
            
            extracted.append((str(zip_file), str(extract_dir)))
            print(f"  Extracted to: {extract_dir}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return extracted


def list_parquet_in_zip(zip_file: str) -> List[str]:
    """List parquet files inside a zip without extracting."""
    with zipfile.ZipFile(zip_file, 'r') as z:
        return [f for f in z.namelist() if f.endswith('.parquet')]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract parquet from zipped GTFS feeds")
    parser.add_argument("--dir", default=".", help="Directory with zip files")
    parser.add_argument("--output", default="extracted_parquet", help="Output directory")
    parser.add_argument("--list", action="store_true", help="Only list contents")
    args = parser.parse_args()
    
    if args.list:
        # List contents of all zip files
        for zip_file in sorted(Path(args.dir).glob("*_files_list.zip")):
            files = list_parquet_in_zip(zip_file)
            print(f"\n{zip_file.name}:")
            for f in files:
                print(f"  {f}")
    else:
        # Extract
        extracted = extract_all_parquet(args.dir, args.output)
        print(f"\nExtracted {len(extracted)} archives")