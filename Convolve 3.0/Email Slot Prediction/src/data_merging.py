"""
Step 1 – Merge communication history with CDNA data using an asof merge.

The asof merge matches each communication record with the most recent CDNA
snapshot that precedes (or equals) its send_timestamp, grouped by customer_code.
Both files are streamed in chunks to stay within memory limits.
"""

import pandas as pd

from configs.config import (
    TRAIN_COMM_FILE, TRAIN_CDNA_FILE, MERGED_TRAIN_FILE,
    TRAIN_CHUNK_SIZE, CDNA_CHUNK_SIZE,
)


def merge_communication_cdna(
    comm_file: str = TRAIN_COMM_FILE,
    cdna_file: str = TRAIN_CDNA_FILE,
    output_file: str = MERGED_TRAIN_FILE,
    comm_chunksize: int = TRAIN_CHUNK_SIZE,
    cdna_chunksize: int = CDNA_CHUNK_SIZE,
) -> None:
    """
    Stream-merge communication history and CDNA data into output_file.

    Strategy
    --------
    For each communication chunk we reload the entire CDNA file in chunks
    and run pd.merge_asof so that every send_timestamp is matched to the
    latest available batch_date for that customer.  Merged chunks are
    appended to output_file to avoid holding everything in RAM.

    Parameters
    ----------
    comm_file     : Path to the raw communication history CSV.
    cdna_file     : Path to the raw CDNA CSV.
    output_file   : Path where the merged output CSV will be written.
    comm_chunksize: Rows per communication chunk (tune to available RAM).
    cdna_chunksize: Rows per CDNA chunk.
    """
    header_written = False

    for comm_chunk in pd.read_csv(comm_file, chunksize=comm_chunksize):
        comm_chunk.columns = [c.lower() for c in comm_chunk.columns]
        comm_chunk["send_timestamp"] = (
            pd.to_datetime(comm_chunk["send_timestamp"]).dt.tz_localize(None)
        )
        comm_chunk["open_timestamp"] = pd.to_datetime(
            comm_chunk["open_timestamp"], errors="coerce"
        )
        comm_chunk = comm_chunk.sort_values("send_timestamp")

        for cdna_chunk in pd.read_csv(cdna_file, chunksize=cdna_chunksize):
            cdna_chunk.columns = [c.lower() for c in cdna_chunk.columns]
            cdna_chunk["batch_date"] = (
                pd.to_datetime(cdna_chunk["batch_date"]).dt.tz_localize(None)
            )
            cdna_chunk = cdna_chunk.sort_values("batch_date")

            merged_chunk = pd.merge_asof(
                comm_chunk,
                cdna_chunk,
                left_on="send_timestamp",
                right_on="batch_date",
                by="customer_code",
                direction="backward",
            )

            merged_chunk.to_csv(
                output_file, index=False, mode="a", header=not header_written
            )
            header_written = True

    print(f"Merged dataset saved to: {output_file}")


if __name__ == "__main__":
    merge_communication_cdna()
