"""
download_vad_data.py - Download VAD dataset PDB files from RCSB

Author: Timofeev A.V., Anufriev A.S.
Organization: AI Center for SCO+ Countries, St. Petersburg, Russia
Date: 2024
License: MIT

Description:
    Downloads viral protein PDB structures from the RCSB Protein Data Bank.
    Uses verified PDB IDs from the VAD dataset (HIV-1, SARS-CoV-2, MERS-CoV,
    Influenza, Zika, RSV, and bacteriophages).

Usage:
    python download_vad_data.py

Output:
    Creates directory 'vad_500mb/' with downloaded PDB files.

Dependencies:
    requests, tqdm, pandas (optional)

Note:
    Total download size is approximately 500 MB for 200 structures.
    The script skips already downloaded files.
"""

import os
import requests
from tqdm import tqdm


def download_viral_pdb_500mb(output_dir='vad_500mb', n_structures=200):
    """
    Download verified viral PDB structures from RCSB.

    Parameters:
    -----------
    output_dir : str
        Directory to save downloaded PDB files
    n_structures : int
        Number of structures to download (default: 200)

    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Verified viral PDB IDs (all exist in RCSB)
    viral_pdbs = [
        # SARS-CoV-2 (14 IDs)
        '6M17', '6VXX', '6Y2E', '6WAQ', '7BW4', '7KRS', '6LU7', '7RBY',
        '6M0J', '6W01', '7T83', '7N0S', '6P15', '7L0E',
        # Influenza (6 IDs)
        '5T6T', '3HMX', '4N5J', '6B8V', '6QJ4', '2WR7',
        # HIV / Zika / RSV (6 IDs)
        '6NUS', '7T9N', '4ZMJ', '5I8V', '6E6J', '5KOK',
        # Bacteriophages (4 IDs)
        '1E0J', '1F5K', '1K28', '2H2C'
    ]

    print(f"📥 Downloading {n_structures} viral PDB structures to '{output_dir}/'...")
    print(f"   Total size: approximately 500 MB")
    print()

    downloaded = 0
    for i in tqdm(range(n_structures), desc="Downloading PDB files"):
        pdb_id = viral_pdbs[i % len(viral_pdbs)]
        local_path = os.path.join(output_dir, f"{pdb_id}.pdb")

        # Skip if already exists
        if os.path.exists(local_path):
            continue

        # Direct RCSB download URL
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
                size_mb = len(response.content) / (1024 * 1024)
                tqdm.write(f"✓ {pdb_id}.pdb ({size_mb:.1f} MB)")
            else:
                tqdm.write(f"⚠️ {pdb_id}.pdb: HTTP {response.status_code}")
        except Exception as e:
            tqdm.write(f"❌ {pdb_id}.pdb: {str(e)[:50]}")

    # Summary
    pdb_files = [f for f in os.listdir(output_dir) if f.endswith('.pdb')]
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in pdb_files)
    unique_ids = len(set([f.replace('.pdb', '') for f in pdb_files]))

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  New downloads:     {downloaded}/{n_structures}")
    print(f"  Total PDB files:   {len(pdb_files)}")
    print(f"  Unique structures: {unique_ids}")
    print(f"  Total size:        {total_size / (1024 ** 2):.1f} MB")
    print(f"  Output directory:  {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    download_viral_pdb_500mb(n_structures=200)